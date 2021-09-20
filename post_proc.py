import os, sys
import numpy as np
import cv2

from xml_parse import parse_xml

def get_bbox_list(anno, source_file_path):
    '''
        input:
            anno: (dict)
                dict_keys(['file_name', 'bbox_list', 'img_size'])
        output:
            bbox_list of source_file_path image
    '''
    target_idx = anno['file_name'].index(source_file_path)
    
    height, width = anno['img_size'][target_idx]
    
    ret = []
    for i in range(len(anno['bbox_list'][target_idx])): # # normalized (x, y, width, height)
        x = int( anno['bbox_list'][target_idx][i][1] * width )
        y = int( anno['bbox_list'][target_idx][i][2] * height) 
        w = int( anno['bbox_list'][target_idx][i][3] * width )
        h = int( anno['bbox_list'][target_idx][i][4] * height)
        ret.append([x, y, h, w])

    return ret

def draw_bbox(img, bbox_list, height, width, class_index=None):
    for box in bbox_list:
        pt1 = (box[1] * width, box[2] * height)
        pt2 = (box[3] * width + pt1[0], box[4] * height + pt1[1])
        
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        cv2.rectangle(img, pt1, pt2, (255,255,0), 4)
        
        if class_index:
            cv2.putText(img, class_index[box[0]], (pt1[0],pt1[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)
    return img


def get_distance(ybr):
    y = ybr

    if y < 216:
        return 8 # meter
    elif y < 532:
        return 6
    elif y < 728:
        return 4
    else:
        return 2

def get_direction(xtl, ytl, xbr, ybr):
    ybr= abs(ybr-1080)
    ytl= abs(ytl-1080)
    
    x = (xbr + xtl)/2
    y = (ybr + ytl)/2

    slope = (x-960)/y
    if 0 <= slope < 0.8:
        return 2
    elif 0.8 <= slope <= 2.5:
        return 1
    elif -0.8 < slope < 0:
        return 10
    elif -2.5 <= slope <= -0.8:
        return 11
    else:
        return 12 


def post_proc(rgb_img, depth_img, bbox_list, class_index):
    '''
        input:
            rgb_img: (ndarray)
            depth_img: (ndarray)
            bbox_list: (list[list[int, int, int, int]])
                list of object bbox 
        output:
            distance: (list[dict{distance: float, direction: int}])
                list of system output (object distance and direction)
    '''
    height, width, _ = rgb_img.shape
    img = rgb_img.copy()
    
    for box in bbox_list: 
        pt1 = (box[1] * width, box[2] * height) 
        pt2 = (box[3] * width + pt1[0], box[4] * height + pt1[1])
        
        pt1 = tuple(map(int, pt1)) # top left 
        pt2 = tuple(map(int, pt2)) # bot right

        distance  = get_distance(pt2[1])
        direction = get_direction(pt1[0], pt1[1], pt2[0], pt2[1])

        cv2.rectangle(img, pt1, pt2, (255,255,0), 4)
        
        cv2.putText(img, class_index[box[0]], (pt1[0],pt1[1]-45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2, cv2.LINE_AA)

        cv2.putText(img, 'dist: {} m'.format(distance), (pt1[0],pt1[1]-25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2, cv2.LINE_AA)

        cv2.putText(img, 'dir: {} o`clock'.format(direction),  (pt1[0],pt1[1]-5),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2, cv2.LINE_AA)
    return img
    
        
if __name__ == '__main__':
    SRC_IMAGE_ROOT='./bbox'
    SRC_DEPTH_ROOT='./depth_result'
    XML_PATH = './bbox/bbox_sample.xml'

    annotation, class_index = parse_xml(SRC_IMAGE_ROOT, XML_PATH)

    for i in range(len(annotation['file_name'])):
        image_name       = annotation['file_name'][i]
        rgb_image_path   = os.path.join(SRC_IMAGE_ROOT, image_name)
        
        depth_image_path = os.path.join(SRC_DEPTH_ROOT, image_name)
        depth_image_path = depth_image_path.replace('jpg', 'png') # jpg -> png

        rgb_img   = cv2.imread(rgb_image_path)      # rgb image
        depth_img = cv2.imread(depth_image_path)    # depth image
        
        print(rgb_img.shape)
        print(depth_img.shape)

        bbox_list = annotation['bbox_list'][i]
        height, width = annotation['img_size'][i]

        # visulization
        '''
        rgb_img   = draw_bbox(rgb_img, bbox_list, height, width, class_index)
        depth_img = draw_bbox(depth_img, bbox_list, height, width, class_index)
        cv2.imwrite('depth_test.png', depth_img)
        cv2.imwrite('rgb_test.png', rgb_img)
        exit()
        '''
        
        # alpha blending
        '''
        alpha = 0.35
        merged = cv2.addWeighted(rgb_img, alpha, depth_img, 1-alpha, 0) 
        out_file_path = 'merged' + str(i+1) + '.png'
        cv2.imwrite(out_file_path, merged)
        '''

        final_img = post_proc(rgb_img, depth_img, bbox_list, class_index)
        out_file_path = 'final' + str(i+1) + '.png'
        cv2.imwrite(out_file_path, final_img)
        
        if i == 2:
            exit()
