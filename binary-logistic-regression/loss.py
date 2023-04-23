import pandas as pd
import numpy as np

# Common loss class
class loss():
    # regularization loss
    def regularization_loss(self, layer):
        # SET DEFAULT REGULARIZATION LOSS = 0
        regularization_loss=0
        # L1 Regularization Weights update
        if layer.weight_regularizer_l1>0:
            regularization_loss+= layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1>0:
            regularization_loss+= layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.bias))
        # L2 Regularization updates
        if layer.weight_regularizer_l2>0:
            regularization_loss+= layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l2>0:
            regularization_loss+= layer.bias_regularizer_l2 * \
                np.sum(layer.bias * layer.bias)
        # return the total regularization loss
        return regularization_loss
    # Calculate sample loss
    def calculate(self, output, y):
        sample_loss=self.forward(output, y)
        ## calculate mean loss
        data_loss=np.mean(sample_loss)
        return data_loss

## loss categorical cross entropy
