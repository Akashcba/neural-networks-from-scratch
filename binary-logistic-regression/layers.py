import numpy as np
import pandas as pd
"""
Code For Dense Layers and Dropout layer from scratch
inspired by Matt Harrison from his book 'Neural Networks From Scratch
"""
## Dense Layer
class layer_dense():
    ## Layer Initialization
    def __init__(self, input_shape, n_neurons,
                weight_regularizer_l1=0, bias_regularizer_l1=0,
                weight_regularizer_l2=0, bias_regularizer_l2=0):
        ### Initialize the weights and bias
        self.weights = .01*np.random.randn(input_shapee, n_neurons)
        self.bais = np.zeros((1,n_neurons))
        ### Set regularizer constatnts
        self.weight_regularizer_l1==weight_regularizer_l1
        self.bias_regularizer_l1=bias_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l2=bias_regularizer_l2
    
    ## Forward Pass
    def forward(self, inputs):
        self.inputs=inputs
        ## Perform Dot Product and sum
        self.output=np.dot(inputs, self.weights) + self.bias
    
    ## Define Backward Pass
    def backward(self, dinputs):
        