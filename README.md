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

"color.tgz", "https://cloud.tsinghua.edu.cn/f/9ce9f2abc61f49ed995a/?dl=1",

"noisy.tgz", "https://cloud.tsinghua.edu.cn/f/674435c8cb914ca0ad10/?dl=1",

"scream.tgz", "https://cloud.tsinghua.edu.cn/f/0613675916ac4c3bb6bd/?dl=1".

MPI3D can be downloaded here:

"real.tgz", "https://cloud.tsinghua.edu.cn/f/04c1318555fc4283862b/?dl=1"),

"realistic.tgz", "https://cloud.tsinghua.edu.cn/f/2c0f7dacc73148cea593/?dl=1",

"toy.tgz", "https://cloud.tsinghua.edu.cn/f/6327912a50374e20af95/?dl=1".

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
