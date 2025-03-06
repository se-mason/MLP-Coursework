import numpy as np
import pandas as pd
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot, correlation_plot

perceptronStructure = ()
learningRate = 0.1
epochs = 10000

# Activation Functions
def get_activation(activation):
    '''Returns the required activation function'''

    # Sigmoid Function
    if activation == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))  
    
def get_activation_derivative(activation):
    '''Returns the derivative of the activation function'''

    # Sigmoid Function
    
# Cost Functions
def cost_function(expectedOutput, outputActivated):
    '''Calculates the error of the neural network'''
    return expectedOutput - outputActivated

# Normalization functions

# Initialise Perceptron Structure
def init_structure(perceptronStructure):
    '''Initialises the perceptron structure'''

    # Split structure
    inputDimensions = perceptronStructure[0]
    nodeDimensions = perceptronStructure[1]

    # Create lists for weights and biases
    weightList = []
    biasList = []

    # Initialise the first layer of the perceptron
    weightMatrix = np.random.uniform(-1, 1, (inputDimensions, nodeDimensions[0][0]))
    biasMatrix = np.random.uniform(-1, 1, (1, nodeDimensions))

    # Append to the lists
    weightList.append(weightMatrix)
    biasList.append(biasMatrix)

    # Initialise the node layers of the perceptron
    for i in range(1, len(nodeDimensions)):
        weightMatrix = np.random.uniform(-1, 1, (nodeDimensions[i - 1][0], nodeDimensions[i][0]))
        biasMatrix = np.random.uniform(-1, 1, (1, nodeDimensions[i][0]))

        # Append to the lists
        weightList.append(weightMatrix)
        biasList.append(biasMatrix)

    return (weightList, biasList)

def forward_pass(inputData, weightList, biasList):
    '''Forward pass of the neural network'''

    # Create a list for the activated nodes and summed nodes
    summedList = []
    activatedList = []


    # Calculate for the hidden layers
    for layer in range(len(weightList)):
        # Get activation function
        activationFunction = get_activation(layer[1])

        # Calculate the sum of the matrix
        sumMatrix = np.dot(inputData, weightList[layer[1]]) + biasList[layer[1]]

        activatedMatrix = activationFunction(sumMatrix)

        # Append to the list
        summedList.append(sumMatrix)
        activatedList.append(activatedMatrix)

        # Update the input data
        inputData = activatedMatrix

    return activatedList, summedList


def backward_pass(expectedOutput, outputActivated, weightList, biasList, summedList, activatedList, inputData, perceptronStructure):
    '''Backward pass of the neural network'''

    # Cost of the network for that pass
    structureCost = cost_function(expectedOutput, outputActivated[-1])

    # Calculate the delta values for the output layer
    get_activation_derivative = get_activation(perceptronStructure[-1][1])
    outputDelta = structureCost * get_activation_derivative(summedList[-1])

    # Calculate the delta values for the hidden layers
    deltaList = []
    deltaList.append(outputDelta)

    # Calculate the delta values for the hidden layers
    for layer in range(len(weightList) - 1, 0, -1):
        delta = (deltaList[-1] @ weightList[layer[1]].T) * get_activation_derivative(summedList[layer[1]])
        deltaList.append(delta)

    # Reverse the delta list
    deltaList.reverse()

    # Calculate the weights and biases with input layer as previous layer
    weightList[0] += (inputData.T @ deltaList[layer[1]]) * learningRate
    biasList[0] += deltaList[layer[1]] * learningRate
    

    # Update the weights and biases for the output layer
    for layer in range(1, len(weightList)+1, 1):
        print(layer)
        weightList[layer[1]] += (activatedList[layer[1]].T @ deltaList[layer[1]]) * learningRate
        biasList[layer[1]] += deltaList[layer[1]] * learningRate

    return (weightList, biasList)