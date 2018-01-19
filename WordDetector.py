import cv2
import numpy as np
import copy
# import Evaluation


class WordDetector:
    _MIN_PTS = 15
    _MAX_PTS = 300
    _MIN_W = 15
    _MAX_W = 40
    _MIN_H = 10#20
    _MAX_H = 100
    _CHANNEL = -1
    _GAU_KSIZE = 5
    _THRESH = 60
    _GAU_SIGMA_X = 0
    _THRESH_MAX_VALUE = 255
    _THRESH_METHOD = 1
    _ELEMENT_SHAPE = 1
    _ELEMENT_KSIZE = 3
    _DILATE_ITER = 4
    _ROI_COL_BEGIN = 23.0 / 100
    _ROI_ROW_BEGIN = 3.0 / 10

    def __init__(self, config = None):
        if config is not None:
            self._MAX_H = config["MAX_H"]
            self._MIN_H = config["MIN_H"]
            self._MAX_W = config["MAX_W"]
            self._MIN_W = config["MIN_W"]
            self._MAX_PTS = config["MAX_PTS"]
            self._MIN_PTS = config["MIN_PTS"]
            self._CHANNEL = config["CHANNEL"]
            self._GAU_KSIZE = config["GAU_KSIZE"]
            self._THRESH = config["THRESH"]
            self._GAU_SIGMA_X = config["GAU_SIGMA_X"]
            self._THRESH_MAX_VALUE = config["THRESH_MAX_VALUE"]
            self._THRESH_METHOD =  config["THRESH_METHOD"]
            self._ELEMENT_KSIZE = config["ELEMENT_KSIZE"]
            self._ELEMENT_SHAPE = config["ELEMENT_SHAPE"]
            self._DILATE_ITER = config["DILATE_ITER"]
            self._ROI_COL_BEGIN = config["ROI_COL_BEGIN"]
            self._ROI_ROW_BEGIN = config["ROI_ROW_BEGIN"]

    def threshold(self, img):
        if self._CHANNEL == -1:
            gray = np.min(img[:,:,:2], axis=2)
        else:
            gray = img[:,:,self._CHANNEL]
            
        gray = cv2.GaussianBlur(gray, (self._GAU_KSIZE, self._GAU_KSIZE), self._GAU_SIGMA_X)
        gray_eq = cv2.equalizeHist(gray)
        binary = cv2.threshold(gray_eq, self._THRESH, self._THRESH_MAX_VALUE, self._THRESH_METHOD)[1]
        # binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 30)

        img_hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
        # mask = cv2.inRange(img_hsv, np.asarray([65, 60, 60]), np.asarray([80, 255, 255]))
        mask = cv2.inRange(img_hsv, np.asarray([65, 100, 50]), np.asarray([80, 255, 255]))
        cv2.imshow("eliminate", mask)
        cv2.imshow("fuck1", binary)
        eliminate = (binary & mask)*255 * 255
        binary = binary - eliminate
        cv2.imshow("fuck2", binary)
        return binary

    def get_boxes(self, binary, delta_x = 0, delta_y = 0):
        # Morphology
        element = cv2.getStructuringElement(self._ELEMENT_SHAPE, (self._ELEMENT_KSIZE, self._ELEMENT_KSIZE))
        _, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow("WTF", binary)
        # for x in range(binary.shape[0]):
        #     for y in range(binary.shape[1]):
        #         if (binary[x][y] != 0 and binary[x][y] != 255): 
        #             print "Hehe ", binary[x][y] 
        dilate = cv2.dilate(binary, element, iterations=1)

        # Drop small and big connected components
        temp = copy.deepcopy(dilate)
        contours = cv2.findContours(temp, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0]
        text_binary = np.zeros(binary.shape, np.uint8)
        for contour in contours:
            x1, y1 = np.min(contour, axis=(0, 1))
            x2, y2 = np.max(contour, axis=(0, 1))
            if y2 - y1 < self._MIN_H :
                continue
            if y2 - y1 > self._MAX_H :
                continue
            cv2.drawContours(text_binary, [contour], -1, 255)
        cv2.imshow("text_binary", text_binary)
        # Morphology
        element[0][1] = 0
        element[2][1] = 0
        cv2.namedWindow("dilate", cv2.cv.CV_WINDOW_NORMAL)

        dilate = cv2.dilate(text_binary, element, iterations=self._DILATE_ITER)
        cv2.imshow("dilate", dilate)


        # Find connected components
        temp = copy.deepcopy(dilate)
        contours = cv2.findContours(temp, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0]
        boxes = []

        for contour in contours:
            x1, y1 = np.min(contour, axis=(0, 1))
            x2, y2 = np.max(contour, axis=(0, 1))
            if y2 - y1 < self._MIN_H:
                continue
            boxes.append([(x1 + delta_x, y1 + delta_y), (x2 + delta_x, y2 + delta_y)])

        return boxes

    def detectWord(self, img):
        ROI = img
        binary = self.threshold(ROI)
        cv2.namedWindow("threshold", cv2.cv.CV_WINDOW_NORMAL)
        cv2.imshow("threshold", binary)
        delta_y = 0
        delta_x = 0
        boxes = self.get_boxes(binary, delta_x, delta_y)
        binary = 255 - binary
        wordImages = []
        boxesImages = [] # x1, y1, w, h
        for box in boxes:
            tmp = [box[0][0],box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]]
            boxesImages.append(tmp)
            wordImages.append(img[box[0][1]: box[1][1], box[0][0]: box[1][0]])
        cv2.namedWindow("step2", cv2.cv.CV_WINDOW_NORMAL)
        cv2.imshow("step2", ROI)
        cv2.waitKey(0)
        return wordImages, boxesImages
        # return boxes