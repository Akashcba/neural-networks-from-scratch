import pandas as pd
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

''' Function calls and training the binary logistic model '''
# load the dataset
x,y=spiral_data(samples=100, classes=2)
# reshape labels to be list of lists
y=y.reshape(-1,1)
# create a dense layer
''' inputs =2, number of neurons = 64 '''
dense_1 = layer_dense(2, 64, weight_regularizer_l2=5e-4,
            bias_regularizer_l2=5e-4)
# Relu activation
activation_1 = activation_relu()
dense_2 = layer_dense(64, 1)
# Sigmoid activation function
activation_2 = activation_sigmoid()
# loss function
loss_function = loss_Binary_Crossentropy()
# create optimizer
optimizer = optimizer_Adam(decay=5e-7)
# Train in loop
for epoch in range(10000):
    # Perform layer 1 forward pass
    dense_1.forward(x)
    # activation 1 pass
    activation_1.forward(dense_1.output)
    # Perform layer 2 foorward pass
    dense_2.forward(activation_1.output)
    # ativation 2 pass
    activation_2.forward(dense_2.output)
    # calculate the loss
    data_loss=loss_functon.calculate(activation_2.output, y)
    # add regularization penalty
    reg_loss = loss_function.regularization_loss(dense_1) + \
               loss_function.regularization_loss(dense_2)
    # total loss
    loss = data_loss + reg_loss
    # Calculate accuracy
    predictions =(activation_2.output > 0.5) *1
    accuracy=np.mean(predictions==y)
    print(f"Epoch {epoch}, Accuracy {accuracy}, Data loss {data_loss}, Reg loss {reg_loss}")
    ''' Perform backward pass '''
    loss_function.backward(activation_2.output, y)
    activation_2.backward(loss_function.dinputs)
    dense_2.backward(activation_2.dinputs)
    activation_1.backward(dense_2.dinputs)
    dense_1.backward(activation_1.dinputs)
    # Perform Parameter updates using the Optimizer
    optimizer.pre_update_params()
    optimizer.update_params(dense_1)
    optimizer.update_params(dense_2)
    optimizer.post_update_params()
# Validate model performance on unseen data
x_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1,1)
''' Make Predictions '''
dense_1.forward(x_test)
activation_1.forward(dense_1.output)
dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)
# Test data loss function
loss = loss_function.calculate(activation_2.output, y_test)
predictions = (activation_2 > 0.5) *1
accuracy = np.mean(predictions==y_test)
print(f"Validation, Accuracy {accuracy}, loss {loss}")
