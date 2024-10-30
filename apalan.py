import cv2
import os
from matplotlib import pyplot as plt 
import numpy as np

sift = cv2.SIFT.create()

index_par = dict(algorithm = 0)
search_par = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_par, search_par)

img_target = cv2.imread('Object.jpg')
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
blur_target = cv2.medianBlur(gray_target, 3)

kp_target, des_target = sift.detectAndCompute(gray_target, None)
des_target=des_target.astype('f')

directory = 'Data'

best_img = None
best_matches = None
best_matches_mask = None
best_kp_img = None
best_total_matches = 0

for i, filename in enumerate (os.listdir(directory)):
    img = cv2.imread(directory + '/' + filename)
    if img is None:
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    
    kp_object, des_object = sift.detectAndCompute(gray, None)
    des_object=des_object.astype('f')
    
    matches = flann.knnMatch(des_object, des_target, 2)
    matchesmask = []
    
    for i in range (0, len(matches)):
        matchesmask.append([0,0])
        
    total_match = 0
    
    for i, (m,n) in enumerate (matches):
        if m.distance < 0.7 * n.distance:
            matchesmask[i] = [1,0]
            total_match += 1
            
    if total_match > best_total_matches:
        best_img = img
        best_kp_img = kp_object
        best_matches = matches
        best_matches_mask = matchesmask
        best_total_matches = total_match
        
img_result=cv2.drawMatchesKnn(
    best_img, best_kp_img, img_target, kp_target,
    best_matches, None,
    matchColor=[0,100,0],
    singlePointColor=[255,0,0],
    matchesMask=best_matches_mask
)

print(best_total_matches)
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.show()