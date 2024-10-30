import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

sift = cv2.SIFT.create()

index_params = dict(algorithm = 0)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

img_target = cv2.imread('Object.jpg')
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

target_median_blur = cv2.medianBlur(gray_target, 3)
# _, target_threshold = cv2.threshold(gray_target, 100, 255, cv2.THRESH_BINARY)

kp_target, des_target = sift.detectAndCompute(target_median_blur, None)
# kp_target, des_target = sift.detectAndCompute(target_threshold, None)

des_target = des_target.astype('f')

directory = 'Data'

# Kalo mau nampilin 1 gambar saja di akhir
best_img = None
best_kp_img = None
best_matches_mask = None
best_matches = None
best_total_matches = 0

for i, filename in enumerate(os.listdir(directory)):
    img = cv2.imread(directory + '/' + filename)
    if img is None:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    median_blur = cv2.medianBlur(gray, 3)
    # _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    kp_object, des_object = sift.detectAndCompute(median_blur, None)
    # kp_target, des_target = sift.detectAndCompute(threshold, None)

    des_object = des_object.astype('f')

    matches = flann.knnMatch(des_object, des_target, 2)
    matchesMask = []
    
    for i in range (0, len(matches)):
        matchesMask.append([0, 0])

    total_match = 0

    for i, (m, n) in enumerate (matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            total_match += 1

    if total_match > best_total_matches:
        best_img = img
        best_kp_img = kp_object
        best_matches = matches
        best_matches_mask = matchesMask
        best_total_matches = total_match

img_result = cv2.drawMatchesKnn(
    best_img, best_kp_img, img_target, kp_target, 
    best_matches, None, 
    matchColor=[0, 100, 0], 
    singlePointColor=[255, 0, 0], 
    matchesMask=best_matches_mask)

print(best_total_matches)
plt.imshow(img_result)
plt.show()