"""
.. module:: DskDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator

class DskDataIterator(DataIterator):

    def __init__(self, file_infos, nb_class, pp_params):
        super().__init__(pp_params)
        self.file_infos = file_infos
        self.nb_class = nb_class

    def init(self, pp_params=None, target_size=(256, 256), color_mode='rgb',
                classes=None, class_mode='categorical',
                batch_size=1, shuffle=True, seed=None,
                save_to_dir=None, save_prefix='', save_format='jpeg',
                follow_links=False         
                ):
            
        super().init(self._imdata_pp.flow_from_directory(self.file_infos, self.nb_class,
                        target_size=target_size, color_mode=color_mode,
                        classes=classes, class_mode=class_mode,
                        batch_size=batch_size, shuffle=shuffle, seed=seed,
                        save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
                        follow_links=follow_links), pp_params)

