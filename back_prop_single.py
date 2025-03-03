import numpy as np


perceptronStructure = (2, 5, 1)
learningRate = 0.1
epochs = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def cost_function(expectedOutput, outputActivated):
    '''Calculates the error of the neural network'''
    return expectedOutput - outputActivated

def init_structure(perceptronStructure):
    '''Intialises the perceptron structure with random weights'''

    # Calculate the first level of the perceptron

    # Weights and biases for input layer to the hidden layer
    hiddenWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[0], perceptronStructure[1]))
    hiddenBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[1]))

    # Weights and biases for hidden layer to the output layer
    outputWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[1], perceptronStructure[2]))
    outputBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[2]))

    # Create tuples for input and output
    hiddenTuple = (hiddenWeightMatrix, hiddenBiasMatrix)
    outputTuple = (outputWeightMatrix, outputBiasMatrix)

    return (hiddenTuple, outputTuple)


def forward_pass(inputData, hiddenTuple, outputTuple):
    '''Forward pass of the neural network'''

    # Calculate for the hidden layer
    hiddenSum = np.dot(inputData, hiddenTuple[0]) + hiddenTuple[1]
    hiddenActivated = sigmoid(hiddenSum)

    # Calculate for the output layer
    outputSum = np.dot(hiddenActivated, outputTuple[0]) + outputTuple[1]
    outputActivated = sigmoid(outputSum)

    return hiddenSum, hiddenActivated, outputSum, outputActivated


def backward_pass(expectedOutput, hiddenSum, hiddenActivated, outputSum, outputActivated, hiddenTuple, outputTuple, inputData):
    '''Backward pass of the neural network'''

    # Calculate the cost of the perceptron structure
    structureCost = cost_function(expectedOutput, outputActivated)

    # Calculate the delta values for the output layer
    outputDelta = structureCost * sigmoid_derivative(outputSum)

    # Calculate the delta values for the hidden layer
    hiddenDelta = np.dot(outputDelta, outputTuple[0].T) * sigmoid_derivative(hiddenSum)


    # Update the weights and biases for the output layer
    outputTuple[0] += np.dot(hiddenActivated.T, outputDelta) * learningRate
    outputTuple[1] += outputDelta * learningRate

    # Update the weights and biases for the hidden layer
    hiddenTuple[0] += np.dot(inputData.T, hiddenDelta) * learningRate
    hiddenTuple[1] += hiddenDelta * learningRate

    return (hiddenTuple, outputTuple)




