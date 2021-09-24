# Restoration-of-Cataract-Images-via-Domain-Adaptation
There is little access to large datasets of cataract images paired with their corresponding clear ones. Therefore, it is unlikely to build a restoration model for cataract images through supervised learning.

Here, we propose an unsupervised restoration method via cataract-like image simulation and domain adaptation.

The code of "An Annotation-free Restoration Network for Cataractous Fundus Images" will be in this repository released soon.

The code of "Domain Generalization in Restoration of Cataract Fundus Images via High-frequency Components" will be in [Restoration-of-Cataract-Images-via-Domain-Generalization](https://github.com/HeverLaw/Restoration-of-Cataract-Images-via-Domain-Generalization).

**Result：**
![Output](images/Output.png)
A comparison of the restored fundus images. (a) cataract image. (b) clear fundus image after surgery. (c) dark channel prior. (d) SGRIF. (e) pix2pix. (f) CycleGAN. (g) the proposed method.

# Prerequisites

\- Win10

\- Python 3

\- CPU or NVIDIA GPU + CUDA CuDNN

# Environment (Using conda)

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing opencv-python

conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9

conda install visdom dominate -c conda-forge # install visdom and dominate

# Simulate cataract-like images

Use the script in ./utils/catacact_simulation.py


# Visualization when training

python -m visdom.server

# To open this link in the browser

http://localhost:8097/

# Dataset preparation

To set up your own dataset constructed like images/cataract_dataset. Note that the number of source images should be bigger than the number of target images.

## Trained model's weight

Download the pretrained model from this link:

https://1drv.ms/u/s!AqcuOey2t2tilG99VRi0h0NSsgu_?e=XFcHOz

Then, place the document in project_root/checkpoints/cataract_model, so that we can get the files like project_root/checkpoints/cataract_model/latest_net_G.pth

# Command to run (root directory is the project root directory)

## train

```
python --dataroot ./datasets/dataset_name --name train_project --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --batch_size 8 --n_epochs 150 --n_epochs_decay 50 --input_nc 6 --output_nc 3```
```

## test

```
python test.py --dataroot ./datasets/dataset_name --name train_project --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --input_nc 6 --output_nc 3
```

## visualization

```
python test.py --dataroot ./images/cataract_dataset --name cataract_model --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --input_nc 6 --output_nc 3
```

