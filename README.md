# Domain-Adaptation-Regression
Code release for Representation Subspace Distance for Domain Adaptation Regression (ICML 2021)

# Domain-Adaptation-Regression

## Prerequisites:

* Python3
* PyTorch == 0.4.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.2.1
* Numpy
* argparse
* PIL

## Dataset:
dSprites can be downloaded here:

"color.tgz", "https://cloud.tsinghua.edu.cn/f/649277b5d5de4c0f8fa2/?dl=1,

"noisy.tgz", "https://cloud.tsinghua.edu.cn/f/35cc1489c7b34ee6a449/?dl=1",

"scream.tgz", "https://cloud.tsinghua.edu.cn/f/583ccf6a795448ec9edd/?dl=1".

MPI3D can be downloaded here:

https://github.com/rr-learning/disentanglement_dataset

Datalists are in the corresponding folder.

## Training on one dataset:

You can reproduce the results by runing rsd.sh in each folder.

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
