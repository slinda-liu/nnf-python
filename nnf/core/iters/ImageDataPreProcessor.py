"""
.. module:: ImageDataGeneratorEx
   :platform: Unix, Windows
   :synopsis: Represent ImageDataGeneratorEx class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports
from keras.preprocessing.image import ImageDataGenerator

# Local Imports
from nnf.core.iters.memory.NumpyArrayIteratorEx import NumpyArrayIteratorEx
from nnf.core.iters.disk.DirectoryIterator import DirectoryIterator
from nnf.core.ImagePreProcessingParam import ImagePreProcessingParam

class ImageDataPreProcessor(ImageDataGenerator):
    def __init__(self, pp_params):

        if (pp_params is None): 
            pp_params = ImagePreProcessingParam()

        super().__init__(featurewise_center=pp_params.featurewise_center,
                        samplewise_center=pp_params.samplewise_center,
                        featurewise_std_normalization=pp_params.featurewise_std_normalization,
                        samplewise_std_normalization=pp_params.samplewise_std_normalization,
                        zca_whitening=pp_params.zca_whitening,
                        rotation_range=pp_params.rotation_range,
                        width_shift_range=pp_params.width_shift_range,
                        height_shift_range=pp_params.height_shift_range,
                        shear_range=pp_params.shear_range,
                        zoom_range=pp_params.zoom_range,
                        channel_shift_range=pp_params.channel_shift_range,
                        fill_mode=pp_params.fill_mode,
                        cval=pp_params.cval,
                        horizontal_flip=pp_params.horizontal_flip,
                        vertical_flip=pp_params.vertical_flip,
                        rescale=pp_params.rescale,
                        preprocessing_function=pp_params.preprocessing_function,
                        dim_ordering=pp_params.dim_ordering)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIteratorEx(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def flow_from_directory(self, file_infos, nb_class,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg',
                            follow_links=False):
        return DirectoryIterator(
            file_infos, nb_class, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            follow_links=follow_links)