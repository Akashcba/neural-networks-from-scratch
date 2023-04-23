import numpy as np
import pandas as pd

''' Optimizers - SGD, AdaGrad, RMSProp, Adam '''
class optimizer_SGD():
    # init the parameters
    def __init__(self, learning_rate=0.001, decay=0, momentum=0):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum
    # Pre Updates func call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=slf.learning_rate*(1/(1+self.decay*self.iterations))
    ## Parameters update functiion
    def update_params(self, layer):
        ## Layer whose parameters are to be  updated
        ''' Check if use momentum'''
        if self.momentum:
            ''' Check if layer contains momentum cache or not '''
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums=np.zeros_like(layer.weights)
                ## Create momentum array for bias as well
                layer.bias_momentums=np.zeros_like(layer.bias)
            ### Create weght updates using momentums - previous_value * retain_factor -> Updat with new gradients
            weight_updates=self.momentum*layer.weight_momentums - selff.current_learning_rate*layer.dweights
            ## Store the updated values in momentum cache
            layer.weight_momentums=weight_updates
            ## Perfform bias updates
            bias_updates=self.momentum*layer.bias_momentums - self.current_learning_rate*layer.dbiases
            ## Store the updates
            layer.bias_momentums=bias_updates
        else:
            ''' Vanilla SGD wiithout momentum '''
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbiases
        ### Perform updates on all parameters
        layer.weights+=weight_updates
        layer.bias+=bias_updates
    ## Post Update Itration counter Function
    def post_update_params(self):
        self.iterations+=1

class optimizer_Adagrad:
    # init the parameters
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
    ## Before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iterations))
    ## Update Parameters
    def update_params(self, layer):
        # Check if layer contains cache arrays
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
        ''' Update cache with squared current gradients '''
        layer.weight_cache+=layer.dweights**2
        layer.bias_cache+=layer.dbiases**2
        # Perform vanilla SGD update with normalization by square rooted cache
        layer.weights += -self.current_learning_rate*layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate*layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    ## Update the iterations count
    def post_update_params(self):
        self.iterations+=1

class RMSProp():
    ## init the parameters
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.rho=rho
    # Before params update function call
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decacy*self.iterations))
    ## Update the parameters
    def update_params(self, layer):
        # create cache arrays if don't exists
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
        # Update cache with squared gradients
        layer.weight_cache=self.rho*layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache=self.rho*layer.bias_cache + (1-self.rho)*layer.dbiases**2
        ## Vanilla SGD update similar to adagrad
        layer.weights+= -self.current_learning_rate*layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias+= -self.current_learning_rate*layer.dbiases / \
                     (np.sqrt(self.bias_cache) + self.epsilon)
        # Update iterations done
    def post_update_params(self):
        self.iterations+=1

class optimizer_Adam():
    # init the parameters
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7,
                beta_1=0.9, beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decacy=decay
        self.epsilon=epsilon
        self.iterations=0
        self.beta_1=beta_1
        self.beta_2=beta_2
    # Call pre update parameters
    def pre_update_params(self):
        if self.decacy:
            self.current_learning_rate=self.learning_rate*1/(1+self.decay*self.iterations)
    ## Update the parameters
    def update_params(self, layer):
        # create cache arrays if does not exists
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums=np.zeros_like(layer.weights)
            layer.weight_cache=np.zeros_like(layer.weights)
            ## Bias
            layer.bias_momentums=np.zeros_like(layer.bias)
            layer.bias_cache=np.zeros_like(layer.bias)

        # Update momentum with current gradients
        layer.weight_momentums=self.beta_1*layer.weight_momentums + \
                               (1-self.beta_1)*layer.dweights
        layer.bias_momentums=self.beat_1*layer.bias_momentums + \
                             (1-self.beta_1)*layer.dbiases
        ### Perform Weight correction for momentum
        weight_momentums_corrected=layer.weight_momentums/ \
            (1-self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected=layer.bias_momentums/ \
            (1-self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache=self.beta_2*layer.weight_cache + \
            (1-self.beta_2)*layer.dweights**2
        layer.bias_cache=self.beta_2*layer.bias_cache + \
            (1-self.beta_2)*layer.dbiases**2
        ## Get corrected cache
        weight_cache_corrected=layer.weight_cache / \
            (1-self.beta_2**(self.iterations + 1))
        bias_cache_corrected=layer.bias_cache / \
            (1-self.beta_2**(self.iterations + 1))
        ### Vanilla Updates with square rooted normalization
        layer.weights+= -self.current_learning_rate*weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.bias += -self.current_learning_rate*bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
    # Update iteration count after successfull updates
    def post_update_params(self):
        self.iterations+=1
