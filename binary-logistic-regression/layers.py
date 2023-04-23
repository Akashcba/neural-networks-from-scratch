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
    def backward(self, dvalues):
        ## Gradients on parameters
        self.dweights=np.dot(self.inputs.T, dvalues)
        self.dbiases=np.sum(dvalues, axis=0, keepdims=True)
        '''Gradients on regularization'''
        ### L1 grad weights
        if self.weight_regularizer_l1>0:
            dl1=np.ones_like(self.weights)
            dl1[self.weights<0]=-1
            self.dweights+=self.weight_regularizer_l1*dl1
        ### L2 grad weights
        if self.bias_regularizer_l2>0:
            self.dweights+=2*self.weight_regularizer_l2*self.weights
        ### L1 Bias Gradients
        if self.bias_regularizer_l1>0:
            dl1=np.ones_like(self.biases)
            dl1[self.biases<0]=-1
            self.dbiases+=self.bias_regularizer_l2*dl1
        ### L2 Bias Gradients
        if self.bias_regularizer_l2>0:
            self.dbiases+=2*self.bias_regularizer_l2*self.biases
        ### Gradients on values
        self.dinputs=np.dot(dvalus, self.weights.T)

### Dropout layer
class layer_dropout():
    # init
    def __init__(self, rate):
        ## Store rate and invert rate with 1 - dropout_rate
        self.rate=1-rate
    ### Forward Pass
    def forward(self,inputs):
        # Save input values
        self.inputs=inputs
        ## Generate and save binary mask
        self.binary_mask=np.random.bionomial(1,self.rate, size=inputs.shape)
        ## Scale the mask
        self.binary_mask/=self.rate
        ## Apply mask on output
        self.output=inputs*self.binary_mask
    ### Backward Pass
    def backward(self, dvalues):
        ''' This layer has no weights an bias params '''
        ### Gradient on values
        self.dinputs=dvalues*self.binary_mask
