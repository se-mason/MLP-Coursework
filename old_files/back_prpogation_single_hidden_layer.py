
import numpy as np



# Define variables
perceptronStructure = (2 ,5 ,1)
learningRate = 0.2
epochs = 1000

# init structure
def init_structure(perceptronStructure):
    '''Initializes the perceptron structure with random weights'''

    # Weights and biases for input layer to the hidden layer
    inputWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[0], perceptronStructure[1]))
    inputBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[1]))

    # Weights and biases for hidden layer to the output layer
    outputWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[1], perceptronStructure[2]))
    outputBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[2]))

    # Create tuples for input and output
    inputTuple = (inputWeightMatrix, inputBiasMatrix)
    outputTuple = (outputWeightMatrix, outputBiasMatrix)

    return (inputTuple, outputTuple)


def forward_pass(inputData, inputTuple, outputTuple):
    '''Forward pass of the neural network'''

    # Initialize the node values matrix
    
    # Calculate for the input layer
    inputSum  = np.dot(inputData, inputTuple[0]) + inputTuple[1]
    inputActivated = sigmoid(inputSum)

    # Calculate for the output layer
    outputSum = np.dot(inputActivated, outputTuple[0]) + outputTuple[1]
    outputActivated = sigmoid(outputSum)

    return inputSum, inputActivated, outputSum, outputActivated


def gradient_descent(expectedOutput, inputSum, inputActivated, outputSum, outputActivated, inputTuple, outputTuple):
    '''Calculates the gradient descent of the neural network'''

    # Calculate the cost of the perceptron structure
    cost = cost_function(expectedOutput, outputActivated)



    



def cost_function(expectedOutput, outputActivated):
    '''Calculates the error of the neural network'''

    # return 0.5 * (expectedOutput - outputActivated) ** 2
    return expectedOutput - outputActivated


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)