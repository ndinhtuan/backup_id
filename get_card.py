import cv2
import numpy as np
import copy
import sys
from perspective import four_point_transform

# input : source img
#output : preprocessed image
def pre_processing(img):

    assert(img is not None, "img is None")
    dst = copy.deepcopy(img)
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    # dst = cv2.inRange(dst, (40, 20, 100), (100, 100, 255))
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # dst = cv2.dilate(dst, element)

    img_hsv = cv2.cvtColor(dst, cv2.cv.CV_BGR2HSV)
    mask = cv2.inRange(img_hsv, np.asarray([60, 20, 20]), np.asarray([100, 255, 255]))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.dilate(mask, element)
    
    return mask

#input : preprocessed_img
#output : box consist card
def find_contour1(preprocessed_img):
    img = copy.deepcopy(preprocessed_img)
    contours, hi = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_TC89_KCOS)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    copy_img = copy.deepcopy(img)
    cv2.drawContours(copy_img, [contours[0]], -1, (255, 0, 0), 3)
    #print "Contour[0]", contours[0]
    cv2.imshow("hj", copy_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("hj")
    rightContour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            rightContour = approx
            break

    return rightContour, copy_img

def find_contour(preprocessed_img):
    img = copy.deepcopy(preprocessed_img)
    contours, hi = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_TC89_KCOS)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    copy_img = copy.deepcopy(img)
    #cv2.drawContours(copy_img, [contours[0]], -1, (255, 0, 0), 3)
    #print "Contour[0]", contours[0]
    # cv2.waitKey(0)
    # cv2.destroyWindow("hj")
    rightContour = None

    peri = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)
    #approx = cv2.convexHull(contours[0])
    # print len(approx), " : ", approx
    # for i in range(len(approx)):
    #     cv2.line(copy_img, (approx[i][0][0], approx[i][0][1]) \
    #                         , (approx[(i+1)%len(approx)][0][0], approx[(i+1)%len(approx)][0][1]), (255, 0, 0), 3)
    rect = cv2.minAreaRect(approx)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(copy_img,[box],-1, (255, 0, 0), 3)

    return box, copy_img

#input : box consist card, src img
#output : perspective card
def get_perspective_card(src_img, box_card):

    assert(box_card is not None)
    pts = box_card.reshape(4,2 )
    warped_img = four_point_transform(src_img, pts)
    return warped_img
