#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import glob
import path
import cv2
import numpy as np
from collections import OrderedDict
import pickle

def main():
    train_path = "./mnist/train/"
    test_path = "./mnist/test/"

    train_paths = glob.glob(train_path + '/*/*')          
    test_paths = glob.glob(test_path + '/*/*')            

    train_dataset = read_image_and_label(train_paths)     
    test_dataset = read_image_and_label(test_paths)       

    save_npy(train_dataset, test_dataset)

    data_dict = read_npy()

    save_pickle(data_dict)

    image = data_dict['train_image'][0]

    data_augment(image)


def read_image_and_label(paths):
    # TODO: with image folders path, read images and make label with image paths)
    # DO NOT use dataset zoo from pytorch or tensorflow
    images = []
    for i in range(len(paths)):
        images.append(cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE))    
        
    labels = []
    for i in range(len(paths)):  
        labels.append(paths[i][-11])            
        
    images = np.array(images)    
    labels = np.array(labels)                 
        
    return images, labels

def save_npy(train_dataset, test_dataset):
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset

    np.save("./train_images.npy",train_images)
    np.save("./test_images.npy", test_images)
    np.save("./train_labels.npy", train_labels)
    np.save("./test_labels.npy", test_labels)

def read_npy():
    # TODO: read npy files and return dictionary
    """
     data = {'train image': [train_images],
             'train label': [train_labels],
             'test_image': [test_images],
             'test_label': [test_labels]
            }
     """
    train_images = np.load('./train_images.npy')
    train_labels = np.load('./train_labels.npy')
    test_images = np.load('./test_images.npy')
    test_labels = np.load('./test_labels.npy')
    
    data_dict = OrderedDict()
    data_dict['train_image'] = train_images
    data_dict['train_label'] = train_labels
    data_dict['test_image'] = test_images
    data_dict['test_label'] = test_labels
    return data_dict

def save_pickle(data_dict):
    # TODO: save data_dict as pickle (erase "return 0" when you finish write your code)
    with open('data_dict','wb') as f:
        pickle.dump(data_dict,f)
#     return 0

def data_augment(image):
    # TODO: use cv2.flip, cv2.rotate, cv2.resize and save each augmented image
    image_flip_TB = cv2.flip(image,0)   
    image_flip_LR = cv2.flip(image,1)   
    image_rotate_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)   
    image_rotate_180 = cv2.rotate(image, cv2.ROTATE_180)
    image_rotate_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_resize = cv2.resize(image,(100,100))
    
    cv2.imwrite('./flip_TB.jpg', image_flip_TB)
    cv2.imwrite('./flip_LR.jpg', image_flip_LR)
    cv2.imwrite('./rotate_90.jpg', image_rotate_90)
    cv2.imwrite('./rotate_180.jpg', image_rotate_180)
    cv2.imwrite('./rotate_270.jpg', image_rotate_270)
    cv2.imwrite('./resize.jpg', image_resize)
    cv2.imwrite('./original.jpg', image)

if __name__ == "__main__":
    main()

