"""MemDataIterator to represent MemDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator

class MemDataIterator(DataIterator):
    """description of class"""

    def __init__(self, nndb, pp_params):
        super().__init__(pp_params)

        # NNdb database to create the iteartor
        self.nndb = nndb

        # Expand to a 4 dimentional database if it is not.
        # flow(...) expect a 4 dimensional database (N x H x W x CH)
        db = self.nndb.db_scipy
        for i in range(self.nndb.db.ndim, 4):
            db = np.expand_dims(db, axis=i)


    def init(self, pp_params=None, batch_size=1, shuffle=True, seed=None,
                save_to_dir=None, save_prefix='', save_format='jpeg', 
                featurewise_center=False,
                samplewise_center=False):

        super().init(self._imdata_pp.flow(self.nndb.db_scipy, y=None, 
                            batch_size=batch_size, shuffle=shuffle, seed=seed,
                            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format), pp_params)