# Restoration-of-Cataract-Images-via-Domain-Adaptation
There is little access to large datasets of cataract images paired with their corresponding clear ones. Therefore, it is unlikely to build a restoration model for cataract images through supervised learning.

Here, we propose an unsupervised restoration method via cataract-like image simulation and domain adaptation, and the code has been released.

The code of "An Annotation-free Restoration Network for Cataractous Fundus Images" , which proposed a model called ArcNet, has been also released.

**Result：**
![Output](images/Output.png)
A comparison of the restored fundus images. (a) cataract image. (b) clear fundus image after surgery. (c) dark channel prior. (d) SGRIF. (e) pix2pix. (f) CycleGAN. (g) the proposed method.

# Prerequisites

\- Win10

\- Python 3

\- CPU or NVIDIA GPU + CUDA CuDNN

# Environment (Using conda)

```
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing opencv-python

conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9

conda install visdom dominate -c conda-forge # install visdom and dominate
```

# Simulate cataract-like images

Use the script in ./utils/catacact_simulation.py


# Visualization when training

python -m visdom.server

# To open this link in the browser

http://localhost:8097/

# Dataset preparation

To set up your own dataset constructed like images/cataract_dataset. Note that the number of source images should be bigger than the number of target images, or you can design you own data loader.

## Trained model's weight

For the model of "Restoration Of Cataract Fundus Images Via Unsupervised Domain Adaptation", please download the pretrained model from this link:

https://drive.google.com/file/d/1Ystqt3RQVfIPPukE7ZdzzFM_hBqB0lr0/view?usp=sharing

Then, place the document in project_root/checkpoints/cataract_model, so that we can get the file like project_root/checkpoints/cataract_model/latest_net_G.pth



For the model of "An Annotation-free Restoration Network for Cataractous Fundus Images", please download the pretrained model from this link:

https://drive.google.com/file/d/1eEzCbKPfKu72UqPBfk3OBUSi-a93T0eg/view?usp=sharing

Then, place the document in project_root/checkpoints/cataract_model, so that we can get the file like project_root/checkpoints/arcnet/latest_net_G.pth

# Command to run

Please note that root directory is the project root directory.

## train

```
python train.py --dataroot ./datasets/dataset_name --name train_project --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --batch_size 8 --n_epochs 150 --n_epochs_decay 50 --input_nc 6 --output_nc 3
```

or

```
python train.py --dataroot ./images/cataract_dataset --name arcnet --model arcnet --netG unet_256 --input_nc 6 --direction AtoB --dataset_mode cataract_guide_padding --norm batch --batch_size 8 --lr_policy step --n_epochs 100 --n_epochs_decay 0 --lr_decay_iters 80 --gpu_ids 0
```

## test & visualization

```
python test.py --dataroot ./datasets/dataset_name --name train_project --model pixDA_sobel --netG unet_256 --direction AtoB --dataset_mode cataract --norm batch --input_nc 6 --output_nc 3
```

or

```
python test.py --dataroot ./images/cataract_dataset --name arcnet --model arcnet --netG unet_256 --input_nc 6 --direction AtoB --dataset_mode cataract_guide_padding --norm batch --gpu_ids 0
```

# Citation

```
@inproceedings{li2021restoration,
  title={Restoration Of Cataract Fundus Images Via Unsupervised Domain Adaptation},
  author={Li, Heng and Liu, Haofeng and Hu, Yan and Higashita, Risa and Zhao, Yitian and Qi, Hong and Liu, Jiang},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={516--520},
  year={2021},
  organization={IEEE}
}
```