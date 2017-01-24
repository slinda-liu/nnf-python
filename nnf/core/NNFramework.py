"""
.. module:: NNFrmework
   :platform: Unix, Windows
   :synopsis: Represent NNFrmework class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports
from abc import ABCMeta, abstractmethod
from warnings import warn as warning

# Local Imports
from nnf.db.DbSlice import DbSlice
from nnf.core.NNDiskMan import NNDiskMan
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.db.Dataset import Dataset
from nnf.core.ImagePreProcessingParam import ImagePreProcessingParam
from nnf.db.Selection import Selection

class NNFramework(object):
    """ABSTRACT CLASS (SHOULD NOT BE INISTANTIATED)"""
    __metaclass__ = ABCMeta

    # Internally used directory to save the processed data
    _SAVE_TO_DIR = "patches"

    def __init__(self, params): 
        pass

    def _process_db_params(self, nnpatches, params):
        """Process db related parameters and attach dbs to corresponding nnpatches."""
        
        # TODO: PERF: Can move this to a one loop
        # Initializing _user_data
        for nnpatch in nnpatches:
            edatasets = Dataset.get_enum_list()
            for edataset in edatasets:
                nnpatch._set_udata(edataset, [])
            
        # Iterate through params
        self._diskmans = []
        i = 0
        for param in params:
            alias = param.setdefault('alias', None)
            nndb = param.setdefault('nndb', None)
            db_dir = param.setdefault('db_dir', None)
            sel = param.setdefault('selection', None)
            pp_param = param.setdefault('db_pp_param', None)
            iter_pp_param = param.setdefault('iter_pp_param', None)
            iter_in_mem = param.setdefault('iter_in_mem', True)

            if ((nndb is not None) and (pp_param is not None) and (iter_pp_param is not None)):
                warning("db_pp_param: ignored, iter_pp_param is used")

            if (sel is None):
                param['selection'] = sel = Selection()

            # Set sel.nnpatches
            if (hasattr(sel, 'nnpatches') and
                sel.nnpatches is not None): warning('ARG_CONFLICT: '
                                                    'sel.nnpatches is already set. '
                                                    'Discarding the current value...')
            sel.nnpatches = nnpatches

            # Slicing the database
            if (iter_in_mem):

                # Warning
                if (nndb is None):
                    warning("""nndb: is not given. 
                                Hence data iterators will not be created.
                                Makesure to use an external DB via NNCfg for training, testing, etc""")
                    continue

                # Split the database according to the selection
                nndbs_tup = DbSlice.slice(nndb, sel, pp_param=pp_param)

                # Update nndbs @ nnpatches
                for pi in range(len(nnpatches)):
                    nnpatch = nnpatches[pi]                    
                    edatasets = nndbs_tup[-1]
                    for ti in range(len(nndbs_tup)-1):                        
                        nndbs = nndbs_tup[ti];
                        if (nndbs is not None): nnpatch._get_udata(edatasets[ti]).append(nndbs[pi])

            else:
                # Initialzie NNDiskman
                if (db_dir is None):
                    warning("""image directory: is not given. 
                        Hence data iterators will not be created.
                        Makesure to use an external DB via NNCfg for training, testing, etc""")
                    dskman = None

                else:
                    # Create diskman and process against the nnpatches
                    save_dir = NNFramework._SAVE_TO_DIR + "_" + (str(i) if (alias is None) else alias)
                    dskman = NNDiskMan(sel, pp_param, nndb, db_dir, save_dir)                                
                    dskman.init()   

                # Update nndbs @ nnpatches
                self._diskmans.append(dskman)

            # Increment the param index variable
            i = i + 1

    def _init_model_params(self, nnpatches, params, nnmodel=None):
        """Initializes models at nnpatches along with iterator stores for 
            the databases stored at nnpatches.
            
            assume self._diskmans is populated.
        """
        # Iterate the patch list
        for nnpatch in nnpatches:

            # Initialize dict|list of iteratorstore for nnmodel @ nnpatch
            # dict_iterstore => [alias1:[iter_TR, iter_VAL, ...], alias2:[iterstore_for_param2_db], ...]
            dict_iterstore = None 

            # list_iterstore => [[iter_TR, iter_VAL, ...], [iterstore_for_param2_db], ...]
            list_iterstore = []
            
            for i in range(len(params)):

                alias = params[i]['alias']
                nndb = params[i]['nndb']
                db_dir = params[i] ['db_dir']
                sel = params[i] ['selection']
                pp_param = params[i] ['db_pp_param']
                iter_pp_param = params[i] ['iter_pp_param']
                iter_in_mem = params[i] ['iter_in_mem']

                iterstore = {}

                # PERF: Create the dictionary in 1st iteration, if only necessary
                if (alias is not None and i==0):
                    dict_iterstore = {}

                if (iter_in_mem):
                    # Create iteartors for the nndb of this param
                    edatasets = Dataset.get_enum_list()
                    for edataset in edatasets:
                        nndbs = nnpatch._get_udata(edataset)

                        memiter = None
                        if (len(nndbs) != 0):
                            memiter =  MemDataIterator(nndbs[i], iter_pp_param)
                            memiter.init()                   
                        
                        iterstore.setdefault(edataset, memiter)

                else:
                    # Create iteartors for the nndb of this param
                    edatasets = Dataset.get_enum_list()                    
                    for edataset in edatasets:
                        dskiter = None
                
                        # Create only when there is a need
                        if ((self._diskmans[i] is not None) and 
                                (self._diskmans[i].get_nb_class(edataset) > 0)):
                            file_infos = self._diskmans[i].get_file_infos(nnpatch.id, edataset)
                            nb_class = self._diskmans[i].get_nb_class(edataset)
                            dskiter =  DskDataIterator(file_infos, nb_class, iter_pp_param)
                            dskiter.init()                   

                        iterstore.setdefault(edataset, dskiter)

                if (alias is not None): dict_iterstore.setdefault(alias, iterstore)
                list_iterstore.append(iterstore)

            # Set the params on the nnmodel (if provided)
            if (nnmodel is not None):
                nnmodel.init_iterstores(dict_iterstore, list_iterstore)
                nnpatch.add_model(nnmodel)

            else:
                # Initialize NN models for this patch
                nnpatch._init_models(dict_iterstore, list_iterstore)
