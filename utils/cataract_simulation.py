# -*- coding: UTF-8 -*-
"""
@Function: simulate the cataract-like images using clean fundus images
@File: cataract_simulation.py
"""
import cv2
import numpy as np
import random
import os
from scipy import ndimage
from PIL import Image, ImageEnhance

# image dir and output image dir
IMAGE_DIR = '../images/original_image'
OUTPUT_DIR = '../images/simulation_image'
# number of cataract-like per clean image
NUM_PER_NOISE = 16


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def gaussian(img):
    kernel_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])
    kernel_5x5 = kernel_5x5 / kernel_5x5.sum()
    k5 = ndimage.convolve(img, kernel_5x5)
    return k5


def cataract_noise(image_name):
    filepath = os.path.join(IMAGE_DIR, image_name)
    im_B = cv2.imread(filepath, 1)

    im_A = cv2.imread(filepath, 1)
    gray_A = cv2.cvtColor(im_A, cv2.COLOR_BGR2GRAY)
    mask_A = ndimage.binary_opening(gray_A > 10, structure=np.ones((8, 8))) * 255

    # get mask
    mask_A_3 = mask_A / mask_A.max()
    mask_A_3 = mask_A_3[:, :, np.newaxis]


    for i in range(NUM_PER_NOISE):
        h, w, c = im_A.shape
        # get random center
        wp = random.randint(int(-w * 0.3), int(w * 0.3))
        hp = random.randint(int(-h * 0.3), int(h * 0.3))
        transmap = np.ones(shape=[h,w])
        # get distance map
        transmap[w // 2 + wp, h // 2 + hp] = 0
        # blur mask
        transmap = gaussian(ndimage.distance_transform_edt(transmap)).T * mask_A
        transmap = transmap / transmap.max()

        sum_map = transmap
        sum_map = (sum_map / sum_map.max())

        # 随机
        randomR = random.choice([3, 5, 7])
        fundus_blur = cv2.GaussianBlur(im_A, (randomR, randomR), 20)
        B, G, R = cv2.split(fundus_blur)
        Rm = np.median(R[R > 5])
        Bm = np.median(B[B > 5])
        Gm = np.median(G[G > 5])

        R = 0.6 * R + sum_map * Rm * 0.8
        G = 0.8 * G + sum_map * Gm * 1
        B = 0.8 * B + sum_map * Bm * 0.8

        sum_degrad = cv2.merge([B, G, R])
        sum_degrad[sum_degrad > 255] = 255

        # color augmentation
        c = random.uniform(0.9, 1.3)
        b = random.uniform(0.9, 1.0)
        e = random.uniform(0.9, 1.3)
        img = Image.fromarray((sum_degrad).astype('uint8'))

        enh_con = ImageEnhance.Contrast(img).enhance(c)
        enh_bri = ImageEnhance.Brightness(enh_con).enhance(b)
        enh_col = ImageEnhance.Color(enh_bri).enhance(e)

        im_A_np = enh_col * mask_A_3

        # save the result
        im_AB = np.concatenate([im_A_np, im_B], 1)
        path_AB = os.path.join(OUTPUT_DIR, image_name.split('.')[0] + '-' + str(i) + '.png')
        cv2.imwrite(path_AB, im_AB)


if __name__ == '__main__':
    image_names = os.listdir(IMAGE_DIR)
    mkdir(OUTPUT_DIR)
    for image_name in image_names:
        cataract_noise(image_name)