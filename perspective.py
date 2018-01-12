import cv2
import numpy as np

def order_points(pts):

    #initialize array with order : top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(img, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    #compute width of wrapped perspective image
    width_top = np.sqrt( (tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    width_bottom = np.sqrt( (bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    max_width = max(int(width_top), int(width_bottom))

    #compute height
    height_left = np.sqrt( (tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    height_right = np.sqrt( (tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    max_height = max(int(height_right), int(height_left))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
        ], dtype="float32")

    mat = cv2.getPerspectiveTransform(rect, dst)
    warped_img = cv2.warpPerspective(img, mat, (max_width, max_height))

    warped_img = cv2.resize(warped_img, (950, max_height * 950/ max_width))

    return warped_img
