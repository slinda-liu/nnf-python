"""DskmanDskDataIterator to represent DskmanDskDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# Local Imports
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator
import nnf.core.NNDiskMan

class DskmanDskDataIterator(DskmanDataIterator):
    """description of class"""

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, db_dir, save_dir, pp_params):
        super().__init__(pp_params)
        
        # Class count, will be updated below
        self.cls_n = 0 
    
        # Keyed by the cls_idx
        # value = [file_path_1, file_path_2, ...] <= list of file paths
        self.paths = {} # A dictionary that hold lists. self.paths[cls_idx] => list of file paths
        
        # Keyed by the cls_idx
        # value = <int> denoting the images per class
        self.n_per_class = {}
        
        # Future use
        self.cls_idx_to_dir = {}       

        # Inner function: Fetch the files in the disk
        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

        # Assign explicit class index for internal reference
        cls_idx = 0

        # Iterate the directory and populate self.paths dictionary
        for root, dirs, files in _recursive_list(db_dir):

            # Exclude this directory itself
            if (root == db_dir):
                continue
            
            # Extract the directory
            dir = root[(root.rindex ('\\')+1):]
        
            # Exclude the internally used data folder
            if (dir == save_dir):
                continue

            # Since dir is considered to be a class name, give the explicit internal index
            self.cls_idx_to_dir.setdefault(cls_idx, dir) # Future use

            # Update class count
            self.cls_n += 1

            # Initialize [paths|n_per_class] dictionaries with related cls_idx
            fpaths = self.paths.setdefault(cls_idx, [])            
            n_per_class = self.n_per_class.setdefault(cls_idx, 0)

            # Update paths
            for fname in files:
                fpath = os.path.join(root, fname)
                fpaths.append(fpath)
                n_per_class += 1
 
            # Update n_per_class dictionary
            self.n_per_class[cls_idx] = n_per_class
            cls_idx += 1

    #################################################################
    # Protected Interface
    #################################################################
    def _get_cimg_in_next(self, cls_idx, col_idx):
        """Fetch image @ cls_idx, col_idx"""
        assert(cls_idx < self.cls_n and col_idx < self.n_per_class[cls_idx])

        impath = self.paths[cls_idx][col_idx]            
        img = load_img(impath, grayscale=False, target_size=None)
        cimg = img_to_array(img, dim_ordering='default')
        return cimg

    def _is_valid_cls_idx(self, cls_idx):
        """Check the validity cls_idx"""
        return cls_idx < self.cls_n        

    def _is_valid_col_idx(self, cls_idx, col_idx):
        """Check the validity col_idx of the class denoted by cls_idx"""
        assert(cls_idx < self.cls_n)
        return col_idx < self.n_per_class[cls_idx]
       
# Sample code
#import os

#def _recursive_list(subpath):
#    return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

#for root, dirs, files in _recursive_list('D:\TestImageFolder'):
#    print(root) 

#for root, dirs, files in _recursive_list('D:\TestImageFolder'):
#    print(dirs) 

#for root, dirs, files in _recursive_list('D:\TestImageFolder'):
#    print(files) 