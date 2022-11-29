

"""Script for extracting ridge ending and bifurcation minutiae patters from fingerprint images."""

import os
from pathlib import Path
import numpy as np
import PIL.Image
import config
import time
import fingerprint_feature_extractor
import cv2 as cv
import pandas as pd

def main():
    # Initialize TensorFlow.
    enhanced_dir = 'enhanced_images'

    pathlist = Path(enhanced_dir).glob('*.png')

    data = []

    for path in pathlist:
        print('file name : ', path)

        img = cv.imread(str(path), 0)

        FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img,
        spuriousMinutiaeThresh=10,
        invertImage = False,
        # showResult=True,
        # saveResult = True
        )

        no_of_re  = len(FeaturesTerminations)
        no_of_rb = len(FeaturesBifurcations)

        data.append([path, no_of_re, no_of_rb])

    df = pd.DataFrame(data, columns=['Filename', 'Ridge Ending', 'Ridge Bifurcation'])
    df.to_csv(enhanced_dir + '/collected_minutiae_stats.csv', header=False, index=False)

if __name__ == "__main__":
    main()
