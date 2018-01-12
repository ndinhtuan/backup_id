# Apply morphology of image : opening, closing, eroding, dilating, ...

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image

def condition_rect(rect, img_shape) :
    x, y, w, h = rect
    (w_img, h_img, _) = img_shape

    threshold_width = 20
    threshold_height = 10
    threshold_S = 2000
    threshold_x = 80
    if w < 1.5 * h : return False
    if h > threshold_height and w * h < (1.0/2) * w_img * h_img and h < (1.0/2) * h_img and y > h_img / 6.5:
        return True

    return False

def compute_area_sum_bound(bound_rects):

    s = 0
    for rect in bound_rects:
        x, y, w, h = rect
        s += w * h
    return s

def resize_box_infor(rect):
    x, y, w, h = rect
    return (x - 5, y, w + 10, h)

def draw_box_infor(img, bound_rects) :
    for rect in bound_rects :
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h),(0, 255, 0), 2)

# function return bounds of information for each threshold value
def dectect_letters(img, thresh_val):

    bound_rects = []
    #threshold directly from src img
    ret, img_preprocess = cv2.threshold(img,thresh_val,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow("img_thresheded", img_preprocess)

    # convert from 3-dim image to 1-dim image
    img_threshold = ((img_preprocess[:, :, 2] > 0) + (img_preprocess[:, :, 1] > 0)) * 255
    img_threshold = img_threshold.astype(np.uint8)
    cv2.imshow("img_threshold", img_threshold)

    #Morphology
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3))
    #img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_GRADIENT, element)
    img_threshold = cv2.dilate(img_threshold, element)
    cv2.imshow("morphology", img_threshold)

    #BOUND INFORmation
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)#cv2.cv.CV_CHAIN_APPROX_TC89_KCOS )

    (w_img, h_img, _) = img.shape
    for contour in contours:
        if len(contour) > 0:
            approx = cv2.approxPolyDP(contour, 2, True)
            cv2.drawContours(img, [approx], -1, [0, 255, 255])
            app_rect = cv2.boundingRect(approx)
            x, y, w, h = app_rect

            #REMOVE Bound, which not consist of infor 's boundary'
            if condition_rect(app_rect, img.shape) is True and cv2.contourArea(approx) >= (1.0/4) * w * h and (x >= w_img / 3):
                app_rect = resize_box_infor(app_rect)
                bound_rects.append(app_rect)

    draw_box_infor(img, bound_rects)
    # draw bounds in image, which coresponds to each threshold value
    cv2.imshow("img_threshold_val", img)
    cv2.waitKey(0)

    return bound_rects

# funtion return bounds of information after choose right threshold
def choose_threshold(img, initial_value) :
    val_thresh = initial_value
    tmp_img = copy.deepcopy(img)
    bound_rects = dectect_letters(tmp_img, val_thresh)
    sum_bounds = compute_area_sum_bound(bound_rects)
    (w_img, h_img, _) = img.shape
    if (sum_bounds == 0):
        return bound_rects
    ratio = w_img * h_img * 1.0 / sum_bounds
    print ratio
    c = 0

    #check suitable for condition of threshold [2.78, 3.4]
    while (3.2 > ratio or ratio > 4.0) and c <= 50:
        c += 1
        tmp_img = copy.deepcopy(img)
        if (3.2 > ratio):
            val_thresh -= 7
        if (4.0 < ratio):
            val_thresh += 5
        bound_rects = dectect_letters(tmp_img, val_thresh)
        sum_bounds = compute_area_sum_bound(bound_rects)
        if (sum_bounds != 0):
            ratio = w_img * h_img * 1.0 / sum_bounds
            print ratio
        else: break
    print "final ratio : {}".format(ratio)

    connect_bounds(bound_rects)
    return bound_rects

#check bound1 and bound2 suitable to connect together
def is_connected(bound1, bound2):

    if (bound1[3] < bound2[3]): # Keep box1's height > box2's height
        tmp = bound1
        bound1 = bound2
        bound2 = tmp

    x1, y1, w1, h1 = bound1
    x2, y2, w2, h2 = bound2

    up_y2 = y2
    down_y2 = y2 + h2

    if (y1 <= up_y2 and up_y2 < y1 + (2.0 / 3) * h1): # box 2 is below box 1
        return True
    if (y1 + h1 >= down_y2 and down_y2 > y1 + (1.0 / 3) * h1): # box 2 is above box 1
        return True
    return False

# connect 2 bound return new bound
def connect(bound1, bound2):
    x1, y1, w1, h1 = bound1
    x2, y2, w2, h2 = bound2

    xmin = min(x1, x2)
    ymin = min(y1, y2)
    xmax = max(x1 + w1, x2 + w2)
    ymax = max(y1 + h1, y2 + h2)

    return (xmin, ymin, xmax - xmin, ymax - ymin) # new x, new y, new w, new h

#function connect 2 bound, which is near together
# return connected bounds
def connect_bounds(bound_rects):
    i = 0

    while i < len(bound_rects) - 1:
        if (is_connected(bound_rects[i], bound_rects[i + 1])):
            #print "BEFORE : {}".format(bound_rects[i])
            bound_rects[i] = connect(bound_rects[i], bound_rects[i + 1])
            #print "AFTER : {}".format(bound_rects[i])
            del bound_rects[i + 1]
        else :
            i += 1
