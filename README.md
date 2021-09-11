# Restoration-of-Cataract-Images-via-Domain-Adaptation
There is little access to large datasets of cataract images paired with their corresponding clear ones. Therefore, it is unlikely to build a restoration model for cataract images through supervised learning.

Here, we propose an unsupervised restoration method via cataract-like image simulation and domain adaptation.

The code of this study will be released soon.

**Result：**
![Output](images/Output.png)
A comparison of the restored fundus images. (a) cataract image. (b) clear fundus image after surgery. (c) dark channel prior. (d) SGRIF. (e) pix2pix. (f) CycleGAN. (g) the proposed method.
## Prerequisites

\- Win10

\- Python 3

\- CPU or NVIDIA GPU + CUDA CuDNN

## Environment (Using conda)

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing opencv-python

conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9

conda install visdom dominate -c conda-forge # install visdom and dominate

## Simulate cataract-like images

Use the script in ./utils/catacact_simulation.py


## Visualization when training

python -m visdom.server

### To open this link in the browser

http://localhost:8097/

## Command to run (root directory is the project root directory)

### train

```
python --dataroot ./datasets/dataset_name --name train_project --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --batch_size 8 --n_epochs 150 --n_epochs_decay 50 --input_nc 6 --output_nc 3```
```

### test

```
python test.py --dataroot ./datasets/dataset_name --name train_project --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --input_nc 6 --output_nc 3
```


# Reference

[1]Jun Cheng et al.,   “Structure-preserving guided retinal image filtering and its application for optic disk analysis,”IEEE TMI, vol. 37, no. 11, pp. 2536–2546, 2018.

[2]Phillip  Isola  et  al.,  “Image-to-image  translation  with conditional adversarial networks,”  in CVPR, 2017, pp.1125–1134.

[3]Yu hao  Luo  et  al.,   “Dehaze  of  cataractous  retinal  images using an unpaired generative adversarial network,”JBHI, 2020.

[4]Ziyi  Shen  et  al.,   “Understanding  and  correcting  low-quality  retinal  fundus  images  for  clinical  analysis,” arXiv preprint arXiv:2005.05594, 2020.

[5]Heng Li et al.,  “Restoration of cataract fundus images via  unsupervised  domain  adaptation,”   in ISBI.  IEEE,2021, pp. 516–520.