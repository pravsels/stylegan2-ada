
"""Script for enhancing fingerprint patterns."""

import os
from pathlib import Path
import numpy as np
import PIL.Image
import config
import time
import fingerprint_enhancer
import cv2 as cv
from tqdm import tqdm

def main():
    # Initialize TensorFlow.
    enhanced_dir = 'enhanced_real_images'

    dataset = 'datasets'

    # create folder if not exist
    os.makedirs(enhanced_dir, exist_ok=True)

    pathlist = [file_path for file_path in Path(dataset).glob('*.png')]

    for path in tqdm(pathlist, total=len(pathlist), desc='Enhancing images'):

        img = cv.imread(str(path), 0)

        out = fingerprint_enhancer.enhance_Fingerprint(img)

        # cv.imshow('enhanced image', out)

        new_path = enhanced_dir + '/' + str(path).split('/')[1]

        cv.imwrite(new_path, out)

if __name__ == "__main__":
    main()
