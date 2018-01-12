from shapely.geometry import Polygon
import numpy as np 

def create_closed_polygon_from_list(list):

    tmp = [i for i in list]
    tmp.append(list[0]) # create closed polygon
    return Polygon(tmp)

def create_polygon_from_ratio(ratio, h_img, w_img):

    ratio_x, ratio_y, ratio_x1, ratio_y1 = ratio

    x = int(w_img / ratio_x)
    y = int(h_img / ratio_y) 
    x1 = int(w_img / ratio_x1)
    y1 = int(h_img / ratio_y1)

    coords = [[x, y], [x1, y], [x1, y1], [x, y1]]
    return create_closed_polygon_from_list(coords)

def create_polygon_from_box(box):
    
    x1, y1, w, h = box 
    x2 = x1+w 
    y2 = y1+h

    coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return create_closed_polygon_from_list(coords)

#box ~ [x0, y0, w, h]
def classify_box(boxes, ratios, h_img, w_img, classified) :

    classified.append([]) #id
    classified.append([]) #name
    classified.append([]) #birth
    classified.append([]) #country
    classified.append([]) #residence registion
    
    polygon_ratios = []
    # create polygon for ratios
    for ratio in ratios :
        polygon_ratios.append(create_polygon_from_ratio(ratio, h_img, w_img))

    # classify each box for fields
    for box in boxes:

        areas_intersections = []
        polygon_box = create_polygon_from_box(box)

        for i in range(len(polygon_ratios)):
            intersections = polygon_box.intersection(polygon_ratios[i]) 
            areas_intersections.append(intersections.area)
        
        index_max = np.argmax(areas_intersections)

        if (areas_intersections[index_max] > 0 and areas_intersections[index_max] > polygon_box.area/2) : # if has one or more field intersect with box
            classified[index_max].append(box)