import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

match_criteria = 500

sd_real_dir = 'sift_descriptors_real/'
sd_synthetic_dir = 'sift_descriptors_synthetic/'

enhanced_dir = 'enhanced_synthetic_images/'
pathlist_synthetic = [file_path for file_path in Path(enhanced_dir).glob('*.png')]

real_dir = 'enhanced_real_images/'
pathlist_real = [file_path for file_path in Path(real_dir).glob('*.png')]

# Initiate SIFT detector
sift = cv.SIFT_create()

sd_real_pathlist = [file_path for file_path in Path(sd_real_dir).glob('*.npy')]
sd_synthetic_pathlist = [file_path for file_path in Path(sd_synthetic_dir).glob('*.npy')]

for real_descriptors_path in tqdm(sd_real_pathlist, total=len(sd_real_pathlist), desc='Looping through real descriptors'):
    real_descriptors = np.load(real_descriptors_path)

    for synthetic_descriptors_path in tqdm(sd_synthetic_pathlist, total=len(sd_synthetic_pathlist), desc='Looping through synthetic data'):

        synthetic_descriptors = np.load(synthetic_descriptors_path)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(synthetic_descriptors, real_descriptors, k=2)

        # Need to draw only good matches, so create a mask
        # matchesMask = [[0,0] for i in range(len(matches))]
        good_matches = 0
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good_matches += 1
                # matchesMask[i]=[1,0]

        if good_matches > match_criteria:
            print('Found match!')
            print('no of good matches : ', good_matches)


# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
#
# print('no of good matches : ', good_matches)
#
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# plt.imshow(img3,),plt.show()
