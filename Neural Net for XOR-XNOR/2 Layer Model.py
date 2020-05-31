__author__ = "Dev Churiwala"

"""
Simple model to predict the XOR/XNOR output of 2 bits based on third input
"""

import numpy as np 


def sigmoid (x):
    return (1/(1 + np.exp(-x)))

def sigmoid_derivative(x):
    return (x * (1 - x))

#Input datasets
inputs = np.array([[1,1,0],[1,0,0],[0,1,0],[0,0,0],[1,1,1],[1,0,1],[0,1,1],[0,0,1]])
expected_output = np.array([[1],[0],[0],[1],[0],[1],[1],[0]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 3,2,1

#Parameter initialization
hidden_weights = np.random.randn(inputLayerNeurons,hiddenLayerNeurons)*0.01
hidden_bias =np.zeros((1,hiddenLayerNeurons))
output_weights = np.random.randn(hiddenLayerNeurons,outputLayerNeurons)*0.01
output_bias = np.zeros((1,outputLayerNeurons))


for _ in range(epochs):
	#Forward Propagation
	hidden_layer_activation = np.dot(inputs,hidden_weights)
	hidden_layer_activation += hidden_bias
	hidden_layer_output = sigmoid(hidden_layer_activation)

	output_layer_activation = np.dot(hidden_layer_output,output_weights)
	output_layer_activation += output_bias
	predicted_output = sigmoid(output_layer_activation)

	#Backpropagation
	error = expected_output - predicted_output
	d_predicted_output = error * sigmoid_derivative(predicted_output)
	
	error_hidden_layer = d_predicted_output.dot(output_weights.T)
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

	#Updating Weights and Biases
	output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
	output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
	hidden_weights += inputs.T.dot(d_hidden_layer) * lr
	hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr


print("Output from neural network after 10,000 epochs: ")
print(*predicted_output)

'''
Output from neural network after 10,000 epochs: 
[0.5000056] [0.50000035] [0.5000069] [0.50000165] [0.49999834] [0.4999931] [0.49999965] [0.4999944]
'''
