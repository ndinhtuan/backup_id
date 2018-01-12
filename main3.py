import cv2
import os
from get_card import pre_processing, find_contour, get_perspective_card
#from get_ratio import on_mouse, get_ratios
import pickle
import copy
import tesserocr
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image, ImageDraw
import time 
from extract_information import dectect_letters, choose_threshold
from get_ratio import try_get_ratios, save_ratios_file, show_ratios_on_files, show_ratios_on_image
from extract_information import connect_bounds
from utils_shape import classify_box
import matplotlib.pyplot as plt

def get_path_file(dir):
    return [os.path.join(dir, i) for i in os.listdir(dir) if os.path.isfile(os.path.join(dir, i))]

#input : image consist card
#output : perspective card, if cannot find contour return None
def get_card_from_image(img):

    dst = pre_processing(img)
    contours, contour_img = find_contour(dst)
    copy_img = copy.deepcopy(img)
    if contours is None:
        print "Cann't Find Contour"
        return None, None, dst, contour_img
    else:
        cv2.drawContours(copy_img, [contours], -1, (255, 0, 0), 4)
    warped_img = get_perspective_card(img, contours)

    return warped_img, copy_img, dst, contour_img

#input : warped_img
#output : coord of box consisting information and image, on which is drawed box
def get_box_information(warped_img, ratios):

    img = copy.deepcopy(warped_img)
    box_infor = []

    h_img, w_img, _ = img.shape

    for [ratio_x, ratio_y, ratio_x1, ratio_y1] in ratios:
        x = int(w_img / ratio_x)
        y = int(h_img / ratio_y)
        x1 = int(w_img / ratio_x1)
        y1 = int(h_img / ratio_y1)
        box_infor.append([x, y, x1, y1])

    return box_infor, img

def threshold(img, channel = 1, ksize = 9, thresh = 35):
    if channel == -1:
        gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    else:
        gray = img[:,:,channel]
    gray = cv2.GaussianBlur(gray, (ksize, ksize), 1)
    gray_eq = cv2.equalizeHist(gray)
    binary =cv2.threshold(gray_eq, thresh, 255, 1)[1]
    return binary

def bound_information(img):

        bound_rects = choose_threshold(img, 100)
        box_information = []
        bounded_img = copy.deepcopy(img)

        for bound in bound_rects:
            x, y, w, h = bound
            box_information.append(img[y-5:y+h, x-5:x+w])
            cv2.rectangle(bounded_img, (x - 5, y - 5), (x+w, y+h),(0, 255, 0), 2)
        return bounded_img, box_information

def save_box_information(img, box_infor, path, starting_number):

    i = starting_number


    for [x, y, x1, y1] in box_infor:

        infor = img[y:y1, x:x1]

        full_path = path + '/' + str(i) + ".jpg"
        cv2.imwrite(full_path, infor)
        i += 1

def find_letters(img, box_infor, path, starting_number) :

    img = threshold(img)
    i = starting_number
    h, w = img.shape

    for [x, y, x1, y1] in box_infor:

        #if (x1 > w or y1 > h): continue
        infor = img[y:y1, x:x1]
        print y1
        print x1
        print h
        print w
        pil_img = Image.fromarray(infor)

        with PyTessBaseAPI() as api:
            api.SetImage(pil_img)
            boxes = api.GetComponentImages(RIL.WORD, True)

            for i, (im, box, _, _) in enumerate(boxes):
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(infor, (x, y), (x + w, y + h), (255, 255, 255))

        full_path = path + '/' + str(i) + ".jpg"
        cv2.imwrite(full_path, infor)
        i += 1
        cv2.imshow("infor_thresh", infor)
        cv2.waitKey(0)

import numpy as np

def testImg(warped_img):
    blueChannel = warped_img[:,:,0]
    greenChannel = warped_img[:, :, 1]
    threshold = 90
    print ((blueChannel > threshold) + (greenChannel > threshold)) * 255
    biChannel = ((blueChannel < threshold) + (greenChannel < threshold)) * 255.0
    print biChannel.shape
    cv2.imshow("testImg", cv2.resize(biChannel, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)

def getInforFromRatios(warped_img, ratios) :

    infor = []
    h, w, _ = warped_img.shape

    for ratio_x, ratio_y, ratio_x1, ratio_y1 in ratios:

        x = int(w / ratio_x)
        y = int(h / ratio_y) 
        x1 = int(w / ratio_x1)
        y1 = int(h / ratio_y1)
        infor.append(warped_img[y:y1, x:x1])
    return infor

from WordDetector import WordDetector

def processAInfor(infor) :

    threshold = 110
    inforBlue = infor[:, :, 0]
    inforGreen = infor[:, :, 1]
    inforBi = ((inforBlue < threshold) + (inforGreen < threshold)) * 255.0
    processedInfor = inforBi
    # wordDetector = WordDetector()
    # processedInfor = wordDetector.threshold(infor)
    return processedInfor

def processInfors(infors):

    processedInfors = [] 

    for infor in infors:
        proInfor = processAInfor(infor)
        processedInfors.append(proInfor)
    
    return processedInfors

def draw_information(warped_img, ratios):

    img = copy.deepcopy(warped_img)

    h_img, w_img, _ = img.shape

    for [ratio_x, ratio_y, ratio_x1, ratio_y1] in ratios:
        
        x = int(w_img / ratio_x)
        y = int(h_img / ratio_y)
        x1 = int(w_img / ratio_x1)
        y1 = int(h_img / ratio_y1)
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0))
        #box_infor.append([x, y, x1, y1])

    return img

def make_box_bigger(boxes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i] 
        x -= 5 #x 
        y -= 5 #y
        w += 60 #w 
        h += 10 #h
        boxes[i] = (x, y, w, h)

def eliminate_not_box(boxes):
    not_box = []

    for i in range(len(boxes)):
        
        if boxes[i][3] * 1.0 / boxes[i][2] > 2: 
            not_box.append(i)

    for i in range(len(not_box)):
        del boxes[not_box[len(not_box) - 1- i]]

def statistic_height_box(boxes):
    tmp = []
    for box in boxes :
        tmp.append(box[3])

    print max(tmp)
    # x_axis = sorted(list(set(tmp)))
    # y_axis = [] 

    # for i in x_axis:
    #     y_axis.append(tmp.count(i))
    
    # y_pos = np.arange(len(y_axis))
    # plt.bar(y_pos, y_axis, alpha=1, align="center")
    # plt.xticks(y_pos, x_axis)
    # plt.ylabel("#number")
    # plt.xlabel("len")
    # plt.show()

# cut boxes into 2 equaly part 
def norm_boxes(boxes):
    threshold_h = 70
    barrier = 3

    for i in range(len(boxes)):
        if boxes[i][3] > threshold_h:
            old_h = boxes[i][3]
            boxes[i] = [boxes[i][0], boxes[i][1], boxes[i][2], old_h /2-barrier] #box1
            box2 = [boxes[i][0], boxes[i][1] + old_h / 2 + barrier, boxes[i][2], old_h/2]
            boxes.insert(i+1, box2)
            i = i + 1

imgs = get_path_file('/home/tuan/Desktop/IdentityCard/SourceImg1')
path_source = '/home/tuan/Desktop/IdentityCard/SourceImg1'

i = -1
haveRatios = True
ratios = None

with open("/home/tuan/Desktop/IdentityCard/ratios.txt", "rb") as fp:
    ratios = pickle.load(fp)

fp.close()

duration = 0

for path_file in imgs:
    print "image " + str(i)
    i += 1
    img = cv2.imread(path_file)
    img1 = copy.deepcopy(img)
    if img is None : 
        print "img is None"
        continue
    t1 = time.time()
    warped_img, copy_img, mask, contour_img = get_card_from_image(img)
    hh = copy.deepcopy(warped_img)
    jj = copy.deepcopy(warped_img)
    t2 = time.time()
    duration += (t2 - t1)

    detectWord = WordDetector()
    wordImages, boxes = detectWord.detectWord(warped_img)

    for box in boxes :
        x, y, w, h = box 
        cv2.rectangle(hh, (x, y), (x+w, y+h), (0, 0, 255))

    # statistic_height_box(boxes)
    eliminate_not_box(boxes)
    norm_boxes(boxes)
    #new norm boxes
    for box in boxes :
        x, y, w, h = box 
        cv2.rectangle(jj, (x, y), (x+w, y+h), (0, 255, 0))

    meaning_boxs = []
    classified = []
    h, w, _ = warped_img.shape
    classify_box(boxes, ratios, h, w, classified)
    for classi, i in zip(classified, range(len(classified))):
        
        connect_bounds(classi)
        # make_box_bigger(classi)
        for box in classi:

            w_img = warped_img.shape[1]
            new_width = w_img / ratios[i][2] - w_img / ratios[i][0]
            box = (box[0], box[1], int(new_width), box[3])
            cv2.rectangle(warped_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
    # statistic_height_box(meaning_boxs)
    tmp = draw_information(warped_img, ratios)
    cv2.imshow("classified", tmp)
    
    cv2.imshow("img", hh)#cv2.resize(img,None, fx=0.3, fy=0.3))
    cv2.imshow("norm", jj)
    cv2.imshow("Warped", warped_img)
    cv2.waitKey(0)

print "Time average = ", duration/22