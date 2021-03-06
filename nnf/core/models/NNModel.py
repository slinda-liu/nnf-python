"""
.. module:: NNModel
   :platform: Unix, Windows
   :synopsis: Represent NNModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports

# Local Imports

# Circular Imports
# ref:http://stackoverflow.com/questions/22187279/python-circular-importing
import nnf.db.NNPatch

class NNModel(object):
    def __init__(self, patches, dict_iterstore, list_iterstore):

        # Initialize instance variables
        self.dict_iterstore = None
        self.list_iterstore = None
        self.patches = []

        # Init iterators store [(Tr, Val, Te), ...]
        self.init_iterstores(dict_iterstore, list_iterstore)

        # Add patch(s)
        if (patches is None): return
        self.add_patches(patches)

    def add_patches(self, patches):
        if (isinstance(patches, list)):
            self.patches = self.patches + patches
        else:
            self.patches.append(patches)

    def init_iterstores(self, dict_iterstore, list_iterstore):
        self.dict_iterstore = dict_iterstore
        self.list_iterstore = list_iterstore

    # USED: in Model based frameworks
    def _init_patches(self):
        self.add_patches(self.generate_patches())

    # USED: in Model based frameworks
    def generate_patches(self):
        """overiddable"""
        nnpatches = []
        nnpatches.append(nnf.db.NNPatch.NNPatch(33, 33, (0, 0)))
        nnpatches.append(nnf.db.NNPatch.NNPatch(33, 33, (10, 10)))
        return nnpatches

    def pre_train(self, daeprecfgs, daecfg):
        """Overiddable"""
        print("TODO: pre_train(daeprecfgs, daecfg): Override and implement")

    def train(self, daecfg):
        """Overiddable"""
        print("TODO: train(daecfg): Override and implement")
