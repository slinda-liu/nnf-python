"""
.. module:: DataIterator
   :platform: Unix, Windows
   :synopsis: Represent DataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports
from abc import ABCMeta, abstractmethod

# Local Imports
from nnf.core.iters.ImageDataPreProcessor import ImageDataPreProcessor

class DataIterator(object):
    """DskmanMemDataIterator represents the diskman iterator for in memory databases.

    Attributes
    ----------
    _gen_data : :obj:`ImageDataGeneratorEx`
        Image generator to apply pre-processing transformations.

    _iter : 

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

    __metaclass__ = ABCMeta

    def __init__(self, pp_params):
        """Constructor of the abstract class DataIterator.

        Parameters
        ----------
        """
        # Initialize the inage data pre-processor with pre-processing params
        self._imdata_pp = ImageDataPreProcessor(pp_params)

        # Core generator (initilaized in self.init() method
        self._gen_next = None

    def init(self, gen_next=None, pp_params=None):
        
        # Set the core generator
        self._gen_next = gen_next

        # Re-init the inage generator with pre-processing params
        if (pp_params is not None):
            self._imdata_pp.__init__(pp_params)        

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y = next(self._gen_next)
        batch_x = batch_x.reshape((len(batch_x), np.prod(batch_x.shape[1:])))
        batch_y = batch_y.reshape((len(batch_y), np.prod(batch_y.shape[1:])))
        return batch_x, batch_y