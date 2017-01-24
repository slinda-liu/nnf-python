"""Selection Module to represent Selection class."""
# -*- coding: utf-8 -*-
# Global Imports
from enum import Enum
import numpy as np

# Local Imports

class Selection:
    """Denote the selection structure.

    Selection Structure (with defaults)
    -----------------------------------
    sel.tr_col_indices      = None    # Training column indices
    sel.tr_noise_mask       = None    # Noisy tr. col indices (bit mask)
    sel.tr_noise_rate       = None    # Rate or noise types for the above field
    sel.tr_out_col_indices  = None    # Training target column indices
    sel.val_col_indices     = None    # Validation column indices
    sel.val_out_col_indices = None    # Validation target column indices
    sel.te_col_indices      = None    # Testing column indices
    sel.nnpatches           = None    # NNPatch object array
    sel.use_rgb             = True    # Use rgb or convert to grayscale
    sel.color_indices       = None    # Specific color indices (set .use_rgb = false)
    sel.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
    sel.scale               = None    # Scaling factor (resize factor)
    sel.normalize           = False   # Normalize (0 mean, std = 1)
    sel.histeq              = False   # Histogram equalization
    sel.histmatch           = False   # Histogram match (ref. image: first image of the class)  # noqa E501
    sel.class_range         = None    # Class range for training database or all (tr, val, te)
    sel.val_class_range     = None    # Class range for validation database
    sel.te_class_range      = None    # Class range for testing database
    sel.pre_process_script  = None    # Custom preprocessing script


    """


    def __init__(self, **kwds):
        """Initialize a selection structure with given field-values."""
        self.tr_col_indices      = None    # Training column indices
        self.tr_noise_mask       = None    # Noisy tr. col indices (bit mask)
        self.tr_noise_rate       = None    # Rate or noise types for the above field
        self.tr_out_col_indices  = None    # Training target column indices
        self.val_col_indices     = None    # Validation column indices
        self.val_out_col_indices = None    # Validation target column indices
        self.te_col_indices      = None    # Testing column indices
        self.nnpatches           = None    # NNPatch object array
        self.use_rgb             = True    # Use rgb or convert to grayscale
        self.color_indices       = None    # Specific color indices (set .use_rgb = false)
        self.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
        self.scale               = None    # Scaling factor (resize factor)
        self.normalize           = False   # Normalize (0 mean, std = 1)
        self.histeq              = False   # Histogram equalization
        self.histmatch           = False   # Histogram match (ref. image: first image of the class)  # noqa E501
        self.class_range         = None    # Class range for training database or all (tr, val, te)
        self.val_class_range     = None    # Class range for validation database
        self.te_class_range      = None    # Class range for testing database
        self.pre_process_script  = None    # Custom preprocessing script
        self.__dict__.update(kwds)

    def __eq__(self, sel):
        """Equality of serilized objects"""

        iseq = False
        if (np.array_equal(self.tr_col_indices, sel.tr_col_indices) and
            np.array_equal(self.tr_noise_mask, sel.tr_noise_mask) and
            np.array_equal(self.tr_noise_rate, sel.tr_noise_rate) and
            np.array_equal(self.tr_out_col_indices, sel.tr_out_col_indices) and
            np.array_equal(self.val_col_indices, sel.val_col_indices) and
            np.array_equal(self.val_out_col_indices, sel.val_out_col_indices) and
            np.array_equal(self.te_col_indices, sel.te_col_indices) and
            (self.use_rgb == sel.use_rgb) and
            np.array_equal(self.color_indices, sel.color_indices) and
            (self.use_real == sel.use_real) and
            (self.normalize == sel.normalize) and
            (self.histeq == sel.histeq) and
            (self.histmatch == sel.histmatch) and
            np.array_equal(self.class_range, sel.class_range) and
            np.array_equal(self.val_class_range, sel.val_class_range) and
            np.array_equal(self.te_class_range, sel.te_class_range) and
            len(self.nnpatches) == len(sel.nnpatches)):
            # self.pre_process_script # LIMITATION: Cannot compare for eqaulity (in the context of serilaization)
            iseq = True

        if (not iseq):
            return iseq

        for i in range(len(self.nnpatches)):
            self_patch = self.nnpatches[i]
            sel_patch = sel.nnpatches[i]
            iseq = iseq and (self_patch == sel_patch)
            if (not iseq):
                break

        return iseq


class Select(Enum):
    """SELECT Enumeration describes the special constants for Selection structure."""

    ALL = -999

    def int(self):
        return self.value