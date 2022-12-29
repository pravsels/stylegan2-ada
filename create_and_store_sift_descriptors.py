import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

store_every = 1

sd_real_dir = 'sift_descriptors_real/'
sd_synthetic_dir = 'sift_descriptors_synthetic/'

enhanced_dir = 'enhanced_synthetic_images/'
pathlist_synthetic = [file_path for file_path in Path(enhanced_dir).glob('*.png')]

real_dir = 'enhanced_real_images/'
pathlist_real = [file_path for file_path in Path(real_dir).glob('*.png')]

# Initiate SIFT detector
sift = cv.SIFT_create()

descriptors = np.empty((0, 128), np.float32)

for j, img2_path in tqdm(enumerate(pathlist_synthetic), total=len(pathlist_synthetic), desc='Looping through dataset'):

    if (j+1) % store_every == 0:
        # print('Shape of accumulated descriptiors : ', descriptors.shape)
        img_filename = str(img2_path).split('/')[-1].split('.')[0]
        np.save(sd_synthetic_dir + img_filename + '_descriptors' + '.npy', descriptors)

        descriptors = np.empty((0, 128), np.float32)

    img2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE) # trainImage
    kp2, des2 = sift.detectAndCompute(img2,None)

    descriptors = np.vstack((descriptors, des2))
