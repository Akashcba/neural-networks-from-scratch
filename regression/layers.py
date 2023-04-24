import pandas as pd
import numpy as np

# Dense Layer
class layer_dense():
    # init
    def __init__(self, n_inputs, n_neurons,
                weight_regularizer_l1=0, bias_regularizer_l1=0,
                weight_regularizer_l2=0, bias_regularizer_l2=0):
        # Glorot initialization of weights
        self.weights= 0.1*np.random.randn(n_inputs, n_neurons)
        self.bias= np.zeros((1, n_neurons))
        ''' Set regularization params'''
        self.weight_regularizer_l1=weight_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l1=bias_regularizer_l1
        self.bias_regularizer_l2=bias_regularizer_l2
    # Forward pass
    def forward(self, inputs):
        # save the inputs
        self.inputs=inputs
        # calculate output
        self.output=np.dot(iinputs, self.weights) + self.bias
    # Backward pass
    def backward(self, dvalues):
        # Get gradients
        self.dweights=np.dot(self.inputs.T, dvalues)
        self.dbiases=np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        ''' L1 regularization grads component '''
        if self.weight_regularizer_l1>0:
            ## dl1 is the sign component
            dl1=np.ones_like(self.weights)
            dl1[self.weights<0]=-1
            # Update Gradients
            self.dweights += self.weight_regularizer_l1 * dl1
        if self.bias_regularizer_l1>0:
            # dl1 is the sgn component
            dl1=np.ones_like(self.bias)
            dl1[self.bias<0]=-1
            # Update gradients
            self.dbiases += self.bias_regularizer_l1 * dl1
        ''' L2 regularization grads component'''
        if self.weight_regularizer_l2>0:
            # Update gradients using the formula
            self.dweights += 2*self.bias_regularizer_l2*self.weights
        if self.bias_regularizer_l2>0:
            # Update gradients
            self.dbiases += 2*self.bias_regularizer_l2*self.bias
        # Save graients on values (based on inputs)
        self.dinputs=np.dot(dvalues, self.weights.T)
    
# Dropout Layer
class layer_dropout():
    # init
    def __init__(self, rate):
        # Store success rate
        self.rate=1-rate
    # forward pass
    def forward(self, inputs):
        # store inputs
        self.inputs=inputs
        # Generate scaled mask
        self.binary_mask=np.randn.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask on output values
        self.output=inputs*self.binary_mask
    # Backward Pass
    def backward(self, dvalues):
        # Gradients on values
        self.dinputs=dvalues*self.binary_mask