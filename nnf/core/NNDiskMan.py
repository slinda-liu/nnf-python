"""NNDiskMan to represent NNDiskMan class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.preprocessing.image import array_to_img
import numpy as np
import pickle
import pprint
import os

# Local Imports
from nnf.core.iters.disk.DskmanDskDataIterator import DskmanDskDataIterator
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice

class NNDiskMan(object):
    """description of class"""

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, sel, pp_params, nndb=None, db_dir=None, save_dir=None):
        
        self.sel = sel
        self.pp_params = pp_params        
        self._save_to_dir = os.path.join(db_dir, save_dir)

        # Create the _save_to_dir if does not exist
        if not os.path.exists(self._save_to_dir):
            os.makedirs(self._save_to_dir)

        # Keyed by the patch_id, and [tr|val|te|etc] dataset key
        # value = (abs_file_patch, cls_lbl) <= file_info tuple
        self.dict_fregistry = {}

        # Keyed by [tr|val|te|etc] dataset key
        # value = <int> denoting the class count
        self.dict_nb_class = {}

        # To save the images that are processed via diskman (default = True)
        self.save_images = True

        # PERF: To update the dictionaries (dict_fregistry, dict_nb_class)
        self.update_dicts = True
    

        # Instantiate an iterator object
        if (nndb is not None):
            self.data_generator = DskmanMemDataIterator(nndb, pp_params)

        elif (db_dir is not None):          
            self.data_generator = None
            create_data_generator = True

            # Try loading the saved diskman
            dskman = self.load_diskman()
            if (dskman is not None and self == dskman):
            
                # PERF: Disable saving of images
                self.save_images = False

                # Fetch the data from the loaded diskman
                self.dict_fregistry = dskman.dict_fregistry
                self.dict_nb_class = dskman.dict_nb_class

                # PERF: Stop updating dictionaries if path are the same
                if (self._save_to_dir == dskman._save_to_dir):
                    self.update_dicts = False
                    create_data_generator = False                    

            if (create_data_generator):
                self.data_generator = DskmanDskDataIterator(db_dir, save_dir, pp_params)

        else:
            raise Exception("NNDiskMan(): Unsupported Mode")
            
    def __eq__(self, diskman):
        """Equality of serilized objects"""

        return (self.sel == diskman.sel) and (self.pp_params == diskman.pp_params)

    def load_diskman(self):
        pkl_fpath = os.path.join(self._save_to_dir, 'diskman.pkl')
        
        if (not os.path.isfile(pkl_fpath)):
            return None

        pkl_file = open(pkl_fpath, 'rb')
        dskman = pickle.load(pkl_file)
        pkl_file.close()

        return dskman

    def save(self):
        pkl_fpath = os.path.join(self._save_to_dir, 'diskman.pkl')
        pkl_file = open(pkl_fpath, 'wb')

        # Pickle the self object using the highest protocol available.
        pickle.dump(self, pkl_file, -1)

        pkl_file.close()


    def init(self):

        # PERF: Update the dictionaries below only if needed
        if (not self.update_dicts):
            return

        # Initialize class ranges and column ranges
        # DEPENDANCY -The order must be preserved as per the enum Dataset 
        # REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4], Refer Dataset enum
        cls_ranges = [self.sel.class_range, self.sel.val_class_range, self.sel.te_class_range]
        col_ranges = [self.sel.tr_col_indices, self.sel.val_col_indices, self.sel.te_col_indices]

        # Set the default range if not specified
        DbSlice._set_default_cls_range(0, cls_ranges, col_ranges)
        self.data_generator.init(cls_ranges, col_ranges)

        # [PERF] Iterate through the choset subset of the disk|nndb database
        for cimg, cls_idx, col_idx, datasets in self.data_generator:             

            # cimg: color/greyscale image 
            # cls_idx, col_idx: int 
            # datasets: list of tuples
            #           [(Dataset.TR, is_new_class), (Dataset.VAL, is_new_class), ...]
    
            # TODO: Use the self.sel structure to process the data

            # Process the patches against cimg
            for nnpatch in self.sel.nnpatches:                
                patch_id = nnpatch.id
                fname = '{cls_idx}_{col_idx}_{patch_id}.{format}'.format(cls_idx=cls_idx,
                                                                    col_idx=col_idx,
                                                                    patch_id=patch_id,
                                                                    format='jpg')
                fpath = os.path.join(self._save_to_dir, fname)
                #fpath = fname # [DEBUG]: comment

                # PERF: Save the images if needed
                if (self.save_images):
                    img = array_to_img(cimg, dim_ordering='default', scale=False)
                    img.save(fpath)                

                for edataset, _ in datasets:
                    self._add_to_fregistry(patch_id, edataset, fpath, cls_idx)
                
            for edataset, is_new_class in datasets:
                if (is_new_class):
                    self._increment_nb_class(edataset)

        # PERF: save the current state of the self object
        self.save()

    def get_file_infos(self, patch_id, ekey):
        """Fetch file info tuples by patch id and [tr|val|te|etc] dataset key"""
        value = self.dict_fregistry.setdefault(patch_id, {})
        file_infos = value.setdefault(ekey, [])
        return file_infos

    def get_nb_class(self, ekey):
        """Fetch class count by [tr|val|te|etc] dataset key""" 
        nb_class = self.dict_nb_class.setdefault(ekey, 0)       
        return nb_class

    #################################################################
    # Private Interface
    #################################################################
    def _increment_nb_class(self, ekey):
        """Add a file info tuple to file registry by patch id and [tr|val|te|etc] dataset key"""
        value = self.dict_nb_class.setdefault(ekey, 0)
        self.dict_nb_class[ekey] = value + 1

    def _add_to_fregistry(self, patch_id, ekey, fpath, cls_lbl):
        """Add a file info tuple to file registry by patch id and [tr|val|te|etc] dataset key"""
        value = self.dict_fregistry.setdefault(patch_id, {})
        value = value.setdefault(ekey, [])
        value.append((fpath, cls_lbl))

    def __getstate__(self):
        """Serialization: Pickle, Remove the following fields from serialization"""
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['data_generator']
        del odict['save_images']
        del odict['update_dicts']
        return odict