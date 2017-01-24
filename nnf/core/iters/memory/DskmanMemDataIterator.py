"""
.. module:: DskmanMemDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanMemDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator
from nnf.core.ImagePreProcessingParam import ImagePreProcessingParam

class DskmanMemDataIterator(DskmanDataIterator):
    """DskmanMemDataIterator represents the diskman iterator for in memory databases.

    Attributes
    ----------
    nndb : :obj:`NNdb`
        Database to iterate.

    Methods
    -------
    init()
        Refer parent class DskmanDataIterator.init(...) method.

    next()
        Refer parent class DskmanDataIterator.next(...) method.

    Examples
    --------
    Construct an iterator, initialize, invoke next().
    >>> iter = DskmanMemDataIterator(nndb)
    >>> iter.init(cls_ranges, col_ranges)
    >>> iter.next()
    """

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, nndb, pp_params):
        """Construct a DskmanMemDataIterator object.

        Parameters
        ----------
        nndb : :obj:`NNdb`
            Database to iterate.
        """
        super().__init__(pp_params)
        self.nndb = nndb

    #################################################################
    # Protected Interface
    #################################################################
    def _get_cimg_in_next(self, cls_idx, col_idx):
        """Fetch image @ cls_idx, col_idx"""
        assert(cls_idx < self.nndb.cls_n and col_idx < self.nndb.n_per_class[cls_idx])

        im_idx = self.nndb.cls_st[cls_idx] + col_idx
        cimg = self.nndb.get_data_at(im_idx)
        return cimg

    def _is_valid_cls_idx(self, cls_idx):
        """Check the validity cls_idx"""
        return cls_idx < self.nndb.cls_n        

    def _is_valid_col_idx(self, cls_idx, col_idx):
        """Check the validity col_idx of the class denoted by cls_idx"""
        assert(cls_idx < self.nndb.cls_n)
        return col_idx < self.nndb.n_per_class[cls_idx]
