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
        ''' Forward pass to be implemented in child class '''
        sample_loss=self.forward(output, y)
        ## calculate mean loss
        data_loss=np.mean(sample_loss)
        return data_loss

## loss categorical cross entropy
class loss_CategoricalCrossentropy(loss):
    # Forward Pass
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        samples=len(y_pred)
        # clip values to prevent division by 0
        # clip bth sides to not drag mean towards any value
        y_pred_clipped=np.clip(y_pred, 1e-7, 1-1e-7)
        # probs for target values
        if len(y_true.shape)==1:
            ''' not one hot encoded '''
            correct_confidence=y_pred_clipped[
                range(samples), y_true
            ]
        if len(y_true.shape)==2:
            ''' One hot encoded '''
            correct_confidence=np.sum( y_pred_clipped * y_true,
                                    axis=1)
            # losses
            negative_log_likelihood= -np.log(correct_confidence)
            return negative_log_likelihood
        # Backward Pass
        def backward(self, dvalues, y_true):
            # Number of samples
            samples=len(dvalues)
            # use first sample to count number of labels in every sample
            labels = len(dvalues[0])
            # if sparse labels turn them into one hot encoder
            if len(y_true.shape)==1:
                y_true=np.eye(labels)[y_true]
            # calculate gradient
            self.dinputs= -y_true / dvalues
            # normalize the gradients
            self.dinputs = self.dinputs/samples

# Categorical Cross Entropy combined with softmax activation
# for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # init the activation and loss objects
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=loss_CategoricalCrossentropy()
    # forward pass
    def forward(self, inputs, y_true):
        # output layers activation function
        self.activation.forward(inputs)
        # set output
        self.output=self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples=len(dvalues)
        ''' If labels are one hot encodded
        turn them into discrete values
        '''
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true, axis=1)
        # copy to modify separately
        self.dinputs=dvalues.copy()
        # calculate gradients
        self.dinputs[range(samples), y_true] -= 1
        # normalize the gradients
        self.dinputs=self.dinputs/samples

# Binary cross Entropy Loss
class loss_Binary_Crossentropy(loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        #clip data to prevent division by 0
        y_pred_clipped=np.clip(y_pred, 1e-7, 1-1e-7)
        # sample wise loss
        sample_loss= -(y_true*np.log(y_pred_clipped) + 
        (1-y_true)*np.log(1-y_pred_clipped))
        sample_loss=np.mean(sample_loss, axis=-1)
        return sample_loss
    # Backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples=len(dvalues)
        # number of outputs in every sample
        outputs=len(dvalues[0])
        # clip data to prevent division by 0
        clipped_dvalues=np.clip(dvalues, 1e-7, 1-1e-7)
        # calculate gradient
        self.dinputs= -(y_true/clipped_dvalues -
        (1-y_true) / (1-clipped_dvalues)) / outputs
        # Normalize the gradient
        self.dinputs=self.dinputs/samples

