# Domain-Adaptation-Regression

## Prerequisites:

* Python3
* PyTorch == 0.4.0/0.4.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.2.1
* Numpy
* argparse
* PIL

## Dataset:

You need to modify the path of the image in every ".txt" in "./data".

## Training on one dataset:

You can use the following commands to the tasks:

python -u train.py --gpu_id n --src src --tgt tgt

n is the gpu id you use, src and tgt can be chosen as in "dataset_list.txt".

## Citation:

If you use this code for your research, please consider citing:

```
@inproceedings{DAR_ICML_21,
  title={Representation Subspace Distance for Domain Adaptation Regression},  
  author={Chen, Xinyang and Wang, Sinan and Wang, Jianmin and Long, Mingsheng}, 
  booktitle={International Conference on Machine Learning}, 
  pages={1749--1759}, 
  year={2021} 
}
```
## Contact
If you have any problem about our code, feel free to contact chenxinyang95@gmail.com.
