from _typeshed import NoneType
import os, sys
import numpy as np
import cv2
from xml.etree.ElementTree import parse

from numpy.lib.type_check import imag
from numpy.lib.utils import byte_bounds

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

def parse_xml(img_dir, xml_path):
    tree = parse(xml_path)
    root = tree.getroot()

    dataset = {'file_name': [], 'bbox_list': [], 'img_size': []}
    
    class_name_dict = {}
    class_index = 0
    class_index_dict = {}

    for child in root.findall('image'):
    # print(child.tag, child.attrib) #image
    # print(child.attrib['id'])
        img_name = os.path.join(img_dir, child.attrib['name'])
        width = int(child.attrib['width'])
        height = int(child.attrib['height'])
        
        dataset['file_name'].append(img_name)
        dataset['img_size'].append((height, width))
        
        bbox_list = []
        for box in child.findall('box'):
            pt1 = (float(box.attrib['xtl']), float(box.attrib['ytl']))
            pt2 = (float(box.attrib['xbr']), float(box.attrib['ybr']))

            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            bbox_width = pt2[0]-pt1[0]
            bbox_height = pt2[1]-pt1[1]

            label = box.attrib['label']

            if not (label in class_name_dict.keys()):
                class_index_dict[class_index] = label
                class_name_dict[label] = class_index
                class_index += 1

            bbox_list.append([class_name_dict[label], pt1[0]/width, pt1[1]/height, bbox_width/width, bbox_height/height])
        
        dataset['bbox_list'].append(bbox_list)
    return dataset, class_index_dict

class AIDogDataset(Dataset):
    def __init__(self, image_dir, xml_path, transform=None, flip=True, jitter=True, blur=True):
        self.samples, self.class_name = parse_xml(image_dir, xml_path)
        self.transform = transform
        if flip:
            self.flip = transforms.RandomHorizontalFlip(p=.5)
        else:
            self.flip = None
        if jitter:
            self.jitter = transforms.ColorJitter(contrast=.005)
        else:
            self.jitter = None
        if blur:
            self.blur = blur
            self.kernel_size = 5
            self.kernel = np.ones((self.kernel_size, self.kernel_size), np.float32) / (self.kernel_size * self.kernel_size)
        else:
            self.blur = None

        assert len(self.samples['file_name']) == len(self.samples['bblox_list'])

        self.transform = transform
        self.samples = []
        self.ground_truth_class = []
        self.ground_truth_bbox = []
        
        word2vec = {'person':torch.tensor[0,0,1]}

        tree = parse(xml)
        root = tree.getroot()

        for child in root.findall('image'):
            img_name = os.path.join(root, child.attrib['name'])
            self.samples.append(img_name)

            bbox_list = [] # [x, y, w, h]
            label_list = []
            for box in child.findall('box'):
                pt1 = (float(box.attrib['xtl']), float(box.attrib['ytl']))
                bbox_list.append(pt1[0])
                bbox_list.append(pt1[1])
                pt2 = (float(box.attrib['xbr']), float(box.attrib['ybr']))
                bbox_list.append(pt2[0] - pt1[0])
                bbox_list.append(pt2[1] - pt1[1])
                
                tmp = copy.deepcopy(word2vec[box.attrib['label']])
                label_list.append(tmp)
                
            self.ground_truth_class.append(tmp)
            self.ground_truth_bbox.append(bbox_list)

    def __len__(self):
        return len(self.samples['file_name'])

    def __getitem__(self, index):
        image = cv2.imread(self.samples['file_name'][index])
        bbox_list = self.samples['bbox_list'][index]   

        # random blur
        if self.blur:
            if np.random.rand() < 0.5:
                img = cv2.filter2D(img, -1, self.kernel)
        
        # transform
        if self.transform:
            img = self.transform(img)

        # flip
        if self.flip:
            img = self.flip(img)
            for i in range(len(bbox_list)):
                if bbox_list[i][1] + bbox_list[i][3] > 0.5:
                    bbox_list[i][1] = 0.5 - abs(bbox_list[i][1]+bbox_list[i][3] - 0.5)
                else:
                    bbox_list[i][1] = 0.5 + abs(bbox_list[i][1]+bbox_list[i][3] - 0.5)

        # color jitter
        if self.jitter:
            if np.random.rand()<0.5:
                img = self.jitter(img)
        
        return (img, bbox_list)

def tensor2numpy(img_tensor):
    img_tensor = ((img_tensor * 0.5) + 0.5).clamp(0.0, 1.0) # -1 ~ 1 -> 0 ~ 1
    np_img = (img_tensor.cpu().detach() * 255.).numpy().astype(np.uint8) # 0 ~ 1 -> 0 ~ 255
    np_img = np_img.transpose(1,2,0) # [:,:,::-1] # C H W -> H W C & bgr -> rgb
    return np_img

def draw_bbox(img_tensor, bbox_list, class_name):
    img = tensor2numpy(img_tensor).copy()
    height, width, _ = img.shape

    # draw bbox
    for bbox in bbox_list:
        obj_type = class_name[bbox[0]]
        bbox = bbox[1:]
        pt1 = (bbox[0] * width, bbox[1] * height)
        pt2 = (pt1[0] + bbox[2] * width, pt1[1] + bbox[3] * height)
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        cv2.rectangle(img, pt1, pt2, (255, 255, 0), 4)
        cv2.putText(img, obj_type, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return img

if __name__ == '__main__':
    IMAGE_DIRECTORY = '.\bbox'
    XML_FILE_PATH = 'C:\\Users\\User\\Desktop\\bbox_sample.xml'

    ai_dog_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_set = AIDogDataset(IMAGE_DIRECTORY, XML_FILE_PATH, transforms=ai_dog_transform, flip=True)

    for idx, sample in enumerate(train_set):
        if idx == 5:
            break
        image, bbox_list = sample

        # debug code
        visualized_img = draw_bbox(image, bbox_list, train_set.class_name)
        file_path = '.\tmp\jitter_'+str(idx+1)+'.png'
        print(f'visualized image has been written in {file_path}.')
        cv2.imwrite(file_path, visualized_img)
