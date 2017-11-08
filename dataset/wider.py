# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os
import numpy as np
from imdb import Imdb
import cv2


class Wider(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    root_path : str
        root path of wider dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, root_path, shuffle=False, is_train=False):
        super(Wider, self).__init__('WIDER_'+image_set)
        self.image_set = image_set
        self.root_path = root_path
        self.is_train = is_train
        self.classes = ['face']
        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()
    
    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.root_path, self.name, 'images.list')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.root_path, self.name, 'images', name)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        anno_path = 'wider_face_{}_info'.format(self.image_set)
        label_file = os.path.join(self.root_path, 'wider_face_split', anno_path, index.replace('.jpg', '.face_bbox'))
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for index in range(self.num_images):
            image_file = self.image_path_from_index(index)
            height, width = self._get_imsize(image_file)
            label_file = self._label_path_from_index(self.image_set_index[index])
            label = []
            with open(label_file) as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    items = [float(x) for x in line.strip().split()]
                    assert len(items) == 4
                    x,y,w,h=items
                    label.append([0, x/width,y/height,(x+w)/width,(y+h)/height])
            temp.append(np.array(label))
        return temp

    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])

if __name__ == '__main__':
    root_path = r'D:\res\face_detect\images\WIDER'
    wider = Wider('val', root_path,True,True)
    print(wider.num_images)
    for idx in range(wider.num_classes):
        image_filename = wider.image_path_from_index(idx)
        label = wider.label_from_index(idx)
        print(image_filename)
        print(label)
        break