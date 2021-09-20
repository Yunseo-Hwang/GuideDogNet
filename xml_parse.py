import os
import cv2
import numpy as np
from xml.etree.ElementTree import parse

# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms

def parse_xml(img_dir, xml_path):
    tree = parse(xml_path)
    root = tree.getroot()

    dataset = {'file_name': [], 'bbox_list': [], 'img_size': []}
    '''
        file_name: [ 1.bmp, 2.bmp, ..., N.bmp]
        img_size: [ (h,w,c), (h,w,c), ... (h,w,c) ]
        bbox_list: [ [[0, x,y,w,h], [1, x,y,w,h], ... ],   # 1.bmp
                     [[0, x,y,w,h], [1, x,y,w,h], ... ],   # 2.bmp
                      ~..                                ] # N.bmp
    '''
    class_name_dict = {} # 'bus': 0.. 
    class_index = 0
    class_index_dict = {}

    for child in root.findall('image'):
        #print(child.tag, child.attrib) # image 
        
        #img_name = os.path.join(img_dir ,child.attrib['name'])
        img_name = child.attrib['name']
        width    = int(child.attrib['width'])
        height   = int(child.attrib['height'])
        
        dataset['file_name'].append(img_name)
        dataset['img_size'].append((height, width))
        #img = cv2.imread(img_name)
        
        bbox_list = []
        for box in child.findall('box'): # bbox in single image
            pt1 = ( float(box.attrib['xtl']), float(box.attrib['ytl']) )
            pt2 = ( float(box.attrib['xbr']), float(box.attrib['ybr']) )

            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            bbox_width  = abs(pt2[0] - pt1[0])
            bbox_height = abs(pt2[1] - pt1[1])
            #img = cv2.rectangle(img, pt1, pt2, (255,0,0), 4)

            lable = box.attrib['label'] # str
            #print(lable)
            if not (lable in class_name_dict.keys()):
                class_index_dict[class_index] = lable # [int] : str
                class_name_dict[lable] = class_index  # [str] : int
                class_index += 1
                

            bbox_list.append([class_name_dict[lable], pt1[0]/width, pt1[1]/height, bbox_width/width, bbox_height/height])

            #cv2.putText(img, lable, ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2) ,cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 1)
        
        dataset['bbox_list'].append(bbox_list)
        #cv2.imwrite('test2.png', img)
    return dataset, class_index_dict