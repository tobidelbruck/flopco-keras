import torch
import torch.nn as nn

import numpy as np
from collections import defaultdict
from functools import partial
import copy
import logging



from flopco_keras.compute_layer_flops import *

log=logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class FlopCoKeras():
    
    def __init__(self, model):
        '''
        instances: list of layer types,
            supported types are [nn.Conv2d, nn.Linear,
            nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Softmax]
        '''
        self.model = model
        
        self.flops = []
        self.macs = []
        self.get_flops = {
                    'ReLU': compute_relu_flops,
                    'InputLayer': compute_input_flops,
                    'Conv2D': compute_conv2d_flops,
                    'ZeroPadding2D': compute_padding_flops,
                    'Activation': compute_activation_flops,
                    'Dense': compute_fc_flops,
                    'BatchNormalization': compute_bn2d_flops,
                    'TensorFlowOpLayer': compute_tfop_flops,
                    'MaxPooling2D': compute_pool2d_flops,
                    'Add': compute_add_flops,
                    'Flatten': compute_flatten_flops,
                    'Dropout': compute_dropout_flops,
                    'GlobalAveragePooling2D': compute_globalavgpool2d_flops,
                    'DepthwiseConv2D': compute_depthwiseconv2d_flops,
                    'Reshape': compute_reshape_flops,
        }

        self.total_flops =None
        self.total_macs = None
        self.relative_flops = None
        self.relative_macs = None

        try:
            self.get_stats( flops = True, macs = True)
        except Exception as e:
            log.error(f'Could not compute flops: caught {e} (probably cannot compute this type of layer)')
            del self.model
            return
        self.total_flops = sum(self.flops)
        self.total_macs = sum(self.macs)
        # self.total_params = sum(self.params) #TO DO 
        
        self.relative_flops = [k/self.total_flops for k in self.flops]
        
        self.relative_macs = [k/self.total_macs for k in self.macs]
        
        # self.relative_params = [k/self.total_params for k in self.params] #TO DO 

        del self.model        
        
    def __str__(self):
        print_info = "\n".join([str({k:v}) for k,v in self.__dict__.items()])
        
        return str(self.__class__) + ": \n" + print_info               
    
    # def count_params(self):
    #     self.params = [0]
        # self.params = defaultdict(int)
        
        # for mname, m in self.model.named_modules():
        #     if m.__class__ in self.instances:
                
        #         self.params[mname] = 0
                
        #         for p in m.parameters():
        #             self.params[mname] += p.numel()
    
    def _save_flops(self, layer, macs=False):
        """ Computes flops for a layer and optionally saves the MACs for that layer

        :param layer: the layer
        :param macs: set True to save layer MACs to self.flops list
        """
        flops = self.get_flops[layer.__class__.__name__](layer, macs)
        if macs:
            self.macs.append(flops)
        else:
            self.flops.append(flops)
        

    def get_stats(self, flops = False, macs = False):
        """ Gets stats for a layer and optionally saves the MACs for that layer
        Use it by constucting f=FlopCo(model), calling f.get_stats(), and then accessing f.total_flops, etc

         :param layer: the layer
         :param macs: set True to save layer MACs to self.flops list

         :returns: None
         """

        # if params:
        #     self.count_params()
       
        if flops:
            self.flops = []
        
        if macs:
            self.macs = []

        layers = None
        if hasattr(self.model, 'layers'):
            log.info('found model.layers')
            layers = self.model.layers
        elif isinstance(self.model, tuple) and hasattr(self.model[0], 'layers'):
            log.info('found model.layers')
            layers = self.model[0].layers
        if flops:
            for layer in layers:
                self._save_flops(layer)
        if macs:
            for layer in layers:
                self._save_flops(layer, macs=True)


