import numpy as np
import pandas as pd

''' Activation functions - Relu, Softmax, Sigmoid'''
# ReLu Activation
class activation_relu():
    # Forward pass
    def forward(self, inputs):
        self.inputs=inputs
        ## Calculate the output
        self.output=np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        self.dinputs=dvalues
        ## Make grad 0 where the value < 0
        self.dinputs[self.inputs<=0]=0
# Sigmoid Activation
class activation_sigmoid():
    # Forward pass
    def forward(self, inputs):
        self.inputs=inputs
        ## Calculate the output
        self.output=1/(1+np.exp(-inputs))
    # Backward pass
    def backward(self, dvalues):
        self.dinputs=dvalues*(1-self.output)*self.output
# Softmax Activation
class activation_softmax():
    # Forward Pass
    def forward(self, inputs):
        self.inputs=inputs
        ## Unnormalized probabilities
        ''' max is used for max value clipping here '''
        exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        ### Normalize the probabilities
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdimsTrue)
        self.output=probabilities
    # Backward pass
    def backward(self, dvalues):
        # Create empty array
        self.dinputs=np.empty_like(dvalues)
        # enumerate output and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten the output array
            single_output=single_output.reshape(-1,1)
            # Jacobian matrix of output
            jacobian_mat=np.diagflat(single_output) - nnp.dot(single_output, single_output.T)
            # Callculate sample wise gradient
            self.dinputs[index]=np.dot(jacobian_mat, single_dvalues)
