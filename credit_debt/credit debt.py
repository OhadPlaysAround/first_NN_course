import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def sigmoid (sum):
    return 1/(1+np.exp(-sum))

def sigmoid_derivative (x):
    return sigmoid(x)*(1-sigmoid(x))

# importing the dataset
dataset = pd.read_csv("C:\\Users\\ohad.benhaim\\Downloads\\credit_data.csv")
#cleaning up the empty cells
dataset = dataset.dropna()
#isolating the inputs and outputs
inputs = dataset.values[:,1:4]
outputs = dataset.values[:,4]
# scaling the inputs
outputs = outputs.reshape(-1, 1)
scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)
# a hidden layer is an average of the inputs (3) and the output (1) so, it should have 2 neurons
# adjusting the weights - 3-by-2 matrix for weight0 and 2-by-1 matrix for weights1
# weights0 = 2 * np.random.random((3, 2)) - 1
# weights1 = 2 * np.random.random((2, 1)) - 1
weights0 = np.zeros((3, 4))
weights1 = np.zeros((4, 1))
epochs = 10000
learning_rate = 0.01
error = []

for epoch in range(epochs):
    input_layer = inputs
    sum_synapse0 = np.dot(input_layer, weights0)
    hidden_layer = sigmoid(sum_synapse0)

    sum_synapse1 = np.dot(hidden_layer, weights1)
    output_layer = sigmoid(sum_synapse1)

    error_output_layer = outputs - output_layer
    average = np.mean(abs(error_output_layer))
    if epoch % 100 == 0:
        print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(average))
        error.append(average)

    derivative_output = sigmoid_derivative(output_layer)
    delta_output = error_output_layer * derivative_output

    weights1T = weights1.T
    delta_output_weight = delta_output.dot(weights1T)
    delta_hidden_layer = delta_output_weight * sigmoid_derivative(hidden_layer)

    hidden_layerT = hidden_layer.T
    input_x_delta1 = hidden_layerT.dot(delta_output)
    weights1 = weights1 + (input_x_delta1 * learning_rate)

    input_layerT = input_layer.T
    input_x_delta0 = input_layerT.dot(delta_hidden_layer)
    weights0 = weights0 + (input_x_delta0 * learning_rate)

print(weights0)
print(weights1)