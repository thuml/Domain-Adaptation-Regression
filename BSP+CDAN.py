import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
torch.set_num_threads(1)
from read_data import ImageList

data_transforms = {
    'train': tran.Visda_train(resize_size=256, crop_size=224),
    'val': tran.Visda_train(resize_size=256, crop_size=224),
    'test': tran.Visda_eval(resize_size=256, crop_size=224),
}
# set dataset
batch_size = {"train": 36, "val": 36, "test": 4}

source_path = "./train_list.txt"
target_path = "./validation_list.txt"

dsets = {"train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
         "val": ImageList(open(target_path).readlines(),transform=data_transforms["val"]),
         "test": ImageList(open(target_path).readlines(),transform=data_transforms["test"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=64)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
dset_classes = range(12)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Visda_test(loader, model,iter_num):
    with torch.no_grad():
        tick = 0
        subclasses_correct = np.zeros(len(dset_classes))
        subclasses_tick = np.zeros(len(dset_classes))
        for (imgs, labels) in loader['test']:
            tick += 1
            imgs = imgs.to(device)
            pred = model(imgs)
            pred = F.softmax(pred)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    subclasses_correct[pred[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print(iter_num)
        for i in range(len(dset_classes)):
            print("\tClass {0} : {1}".format(i, subclasses_result[i]))
        print("\tAvg : {0}\n".format(subclasses_result.mean()))



def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer



class BSP_CDAN(nn.Module):
    def __init__(self, num_features):
        super(BSP_CDAN, self).__init__()
        self.model_fc = model.Resnet101Fc()
        self.bottleneck_layer1 = nn.Linear(num_features, 256)
        self.bottleneck_layer1.apply(init_weights)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = nn.Linear(256, len(dset_classes))
        self.classifier_layer.apply(init_weights)
        self.predict_layer = nn.Sequential(self.model_fc, self.bottleneck_layer, self.classifier_layer)

    def forward(self, x):
        feature = self.model_fc(x)
        out = self.bottleneck_layer(feature)
        outC = self.classifier_layer(out)
        return (out, outC)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def srpolicy(iter_num, max_iter, init_lr=0.005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = (iter_num / max_iter) * 4 * init_lr + init_lr
    return lr

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


num_features = 2048
DANN = BSP_CDAN(num_features)
DANN = DANN.to(device)
ad_net = AdversarialNetwork(256 * len(dset_classes), 1024)
ad_net = ad_net.to(device)
DANN.train(True)
ad_net.train(True)
criterion = {"classifier": nn.CrossEntropyLoss(), "adversarial": nn.BCELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, DANN.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, DANN.bottleneck_layer.parameters()), "lr": 1},
                  {"params": filter(lambda p: p.requires_grad, DANN.classifier_layer.parameters()), "lr": 1},
                  {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": 1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])
for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = 500
num_iter = 40002
for iter_num in range(1, num_iter + 1):
    DANN.train(True)
    optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=0.003, gamma=0.0001, power=0.75,
                                 weight_decay=0.0005)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["val"])
    data_source = iter_source.next()
    data_target = iter_target.next()
    inputs_source, labels_source = data_source
    inputs_target, labels_target = data_target
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    dc_target = torch.from_numpy(np.array([[1], ] * batch_size["train"] + [[0], ] * batch_size["train"])).float()
    inputs = inputs.to(device)
    labels = labels_source.to(device)
    dc_target = dc_target.to(device)
    feature, outC = DANN(inputs)
    feature_s = feature.narrow(0, 0, int(feature.size(0) / 2))
    feature_t = feature.narrow(0, int(feature.size(0) / 2), int(feature.size(0) / 2))
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    sigma_loss = 0.0001 * sigma
    classifier_loss = criterion["classifier"](outC.narrow(0, 0, batch_size["train"]), labels)
    total_loss = classifier_loss
    softmax_out = nn.Softmax(dim=1)(outC)
    entropy = Entropy(softmax_out)
    coeff = calc_coeff(iter_num)
    transfer_loss = CDAN([feature, softmax_out], ad_net, entropy, coeff, random_layer=None)
    total_loss = total_loss + transfer_loss + sigma_loss
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    train_transfer_loss += transfer_loss.item()
    train_total_loss += total_loss.item()
    train_sigma += sigma_loss.item()
    if iter_num % test_interval == 0:
        print(
        "Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f}; Average Sigma Loss: {:.4f}; Average Training Loss: {:.4f}".format(
            iter_num, train_cross_loss / float(test_interval), train_transfer_loss / float(test_interval),
                      train_sigma / float(test_interval),
                      train_total_loss / float(test_interval)))
        train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
    if (iter_num % 1000) == 0:
        DANN.eval()
        Visda_test(dset_loaders, DANN.predict_layer,iter_num)