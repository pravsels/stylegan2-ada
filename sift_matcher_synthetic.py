import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

match_criteria = 500

enhanced_dir = 'enhanced_synthetic_images/'

pathlist = [file_path for file_path in Path(enhanced_dir).glob('*.png')]

print('no of images in the folder : ', len(pathlist))

# Initiate SIFT detector
sift = cv.SIFT_create()

for i, img1_path in tqdm(enumerate(pathlist), total=len(pathlist), desc='Checking for duplicates'):
    img1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE) # queryImage
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)

    for j, img2_path in enumerate(pathlist):
        if img1_path == img2_path:
            continue

        img2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE) # trainImage
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

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

# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
#
# print('no of good matches : ', good_matches)
#
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# plt.imshow(img3,),plt.show()
