# back Propogation algorithm
import numpy as np

perceptronStructure = (2,[(2,'sigmoid'),(3,'sigmoid'),(2,'sigmoid'), (1, 'sigmoid')])
learningRate = 0.1

#re do this bit for sure

def get_activation_function(name):
    """
    Returns an activation function that works with both np.matrix and np.ndarray.
    """
    activations = {
        "relu": lambda x: np.maximum(0, np.asarray(x)),  
        "sigmoid": lambda x: 1 / (1 + np.exp(-np.asarray(x))),  
        "tanh": lambda x: np.tanh(np.asarray(x)),  
        "linear": lambda x: np.asarray(x),  
        "softmax": lambda x: np.exp(np.asarray(x)) / np.sum(np.exp(np.asarray(x)))  
    }

    return activations.get(name, lambda x: np.asarray(x))  # Default to linear

def get_activation_derivative(name):
    """
    Returns the derivative of the given activation function.
    """
    derivatives = {
        "relu": lambda x: np.where(x > 0, 1, 0),  # Derivative of ReLU
        "sigmoid": lambda x: (s := 1 / (1 + np.exp(-x))) * (1 - s),  # Sigmoid derivative: s * (1 - s)
        "tanh": lambda x: 1 - np.tanh(x) ** 2,  # Derivative of tanh: 1 - tanh^2(x)
        "linear": lambda x: np.ones_like(x),  # Derivative of linear (identity) function is 1
        "softmax": lambda x: (s := np.exp(x) / np.sum(np.exp(x))) * (1 - s)  # Softmax derivative (simplified)
    }

    return derivatives.get(name, lambda x: np.ones_like(x))  # Default to 1 if not found


def init_structure(perceptronStructure):
    '''Initializes the perceptron structure with random weights'''

    weightsList = []
    biasList = []

    inputWeightMatrix = np.random.uniform(1, 1, (perceptronStructure[0], perceptronStructure[1][0][0]))
    weightsList.append(inputWeightMatrix)

    inputBiasMatrix = np.random.uniform(1, 1, (1, perceptronStructure[1][0][0]))
    biasList.append(inputBiasMatrix)

    for i in range(1, len(perceptronStructure[1])):
        hiddenWeightMatrix = np.random.uniform(1, 1, (perceptronStructure[1][i-1][0], perceptronStructure[1][i][0]))
        weightsList.append(hiddenWeightMatrix)


        hiddenBiasMatrix = np.random.uniform(1, 1, (1, perceptronStructure[1][i][0]))
        biasList.append(hiddenBiasMatrix)

    return (weightsList, biasList)


    

def forward_pass(data, weightsAndBiases, perceptronStructure):
    '''Forward pass of the neural network'''

    # Initialize the node values matrix
    nodeValues = []
    # Prepare the matrices for calculation
    weightsList, biasList = weightsAndBiases

    # Calculate values at the first hidden layer
    activation = get_activation_function(perceptronStructure[1][0][1])
    previousLayer = activation(np.dot(data, weightsList[0]) + biasList[0])
    nodeValues.append(previousLayer)

    # Calculate values at the rest of the hidden layers
    for i in range(1, len(weightsList)):
        # Get this layers activation function
        activation = get_activation_function(perceptronStructure[1][i][1])
        # Define the current hidden layer with matrix calculations
        previousLayer = activation(np.dot(previousLayer, weightsList[i]) + biasList[i])
        # Append the current hidden layer to the node values matrix
        nodeValues.append(previousLayer)

    return nodeValues

def delta_function(outputMatrix,currentStructure, dataOutput, learningRate):
    '''Calculate the delta function'''
    # Get the inverse activation function
    activation = get_activation_derivative(currentStructure[1])
    inverseOutput = activation(outputMatrix).flatten()

    vectorForm = outputMatrix.flatten()

    # turn the data output into a vector
    dataOutput = dataOutput * np.ones(vectorForm.shape)

    # Calculate the delta function
    delta = (dataOutput - vectorForm) * inverseOutput
    delta.reshape(outputMatrix.shape)

    return delta


def backward_pass(nodeValues, weightsAndBiases, dataOutput, learningRate, perceptronStructure):
    '''back pass algorithm'''

    weightsList, biasList = weightsAndBiases

    for i in range(0, len(nodeValues)):
        matrixPosition = (-(i+1))
        # get the current working matrix 
        workingMatrix = nodeValues[matrixPosition]
        workingWeights = weightsList[matrixPosition]
        currentStructure = perceptronStructure[1][matrixPosition]

        delta = delta_function(workingMatrix, currentStructure, dataOutput, learningRate)
        print(f'delta function: {delta} for pos {matrixPosition}')
        # back pass algor




test_data = np.random.uniform(2,2, (1, 2))
print(test_data)

weightsAndBiases = init_structure(perceptronStructure)

nodeValues = (forward_pass(test_data, weightsAndBiases, perceptronStructure))
print(nodeValues)
dataOutput = (3)
backward_pass(nodeValues, weightsAndBiases, dataOutput, learningRate, perceptronStructure)