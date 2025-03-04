import numpy as np
import pandas as pd
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot


perceptronStructure = (3, 9, 1)
learningRate = 0.12
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def cost_function(expectedOutput, outputActivated):
    '''Calculates the error of the neural network'''
    return expectedOutput - outputActivated


def min_max_scaler(data):
    '''Function to normalize the data'''
    return (data - data.min()) / (data.max() - data.min())



def init_structure(perceptronStructure):
    '''Intialises the perceptron structure with random weights'''

    # Calculate the first level of the perceptron

    # Weights and biases for input layer to the hidden layer
    hiddenWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[0], perceptronStructure[1]))
    hiddenBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[1]))

    # Weights and biases for hidden layer to the output layer
    outputWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[1], perceptronStructure[2]))
    outputBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[2]))

    # Create lists for input and output
    hiddenList = [hiddenWeightMatrix, hiddenBiasMatrix]
    outputList = [outputWeightMatrix, outputBiasMatrix]

    return (hiddenList, outputList)


def forward_pass(inputData, hiddenList, outputList):
    '''Forward pass of the neural network'''

    # Calculate for the hidden layer
    hiddenSum = np.dot(inputData, hiddenList[0]) + hiddenList[1]
    hiddenActivated = sigmoid(hiddenSum)

    # Calculate for the output layer
    outputSum = np.dot(hiddenActivated, outputList[0]) + outputList[1]
    outputActivated = sigmoid(outputSum)

    return hiddenSum, hiddenActivated, outputSum, outputActivated


def backward_pass(expectedOutput, hiddenSum, hiddenActivated, outputSum, outputActivated, hiddenList, outputList, inputData):
    '''Backward pass of the neural network'''

    # Calculate the cost of the perceptron structure
    structureCost = cost_function(expectedOutput, outputActivated)

    # Calculate the delta values for the output layer
    outputDelta = structureCost * sigmoid_derivative(outputSum)

    # Calculate the delta values for the hidden layer
    hiddenDelta = np.dot(outputDelta, outputList[0].T) * sigmoid_derivative(hiddenSum)


    # Update the weights and biases for the output layer
    outputList[0] += np.dot(hiddenActivated.T, outputDelta) * learningRate
    outputList[1] += outputDelta * learningRate

    # Update the weights and biases for the hidden layer
    hiddenList[0] += np.dot(inputData.T, hiddenDelta) * learningRate
    hiddenList[1] += hiddenDelta * learningRate

    return (hiddenList, outputList)


def train_network(hiddenList, outputList):
    '''Trains the neural network with the training data'''
    count = 0

    # Load the data
    dataBase = rw.read_data_all('dataSet.db', 'data_table')
    dataBase.iloc[:, 1:] = min_max_scaler(dataBase.iloc[:, 1:])

    # Initialize an empty DataFrame to store errors
    errorDF = pd.DataFrame(columns=['epoch', 'error'])

    while count < epochs:
        predictions = []
        for i, day in enumerate(dataBase.itertuples(index=False), start=0):
            # No errors for our of range datapoints (first and last)
            if i == len(dataBase)-1 or i == 0:
                continue

            # Get the previous and next day
            nextDay = tuple(dataBase.iloc[i + 1])
            previousDay = tuple(dataBase.iloc[i - 1])

            inputData = np.array([float(day[3]), float(day[8]), float(previousDay[7])])
            expectedOutput = np.array(float(nextDay[4]))

            # Forward pass
            hiddenSum, hiddenActivated, outputSum, outputActivated = forward_pass(inputData, hiddenList, outputList)

            # Backward pass
            hiddenList, outputList = backward_pass(expectedOutput, hiddenSum, hiddenActivated, outputSum, outputActivated, hiddenList, outputList, inputData.reshape(1, -1))


            predictions.append({'DATE': nextDay[0], 'Skelton': outputActivated[0]})

        count += 1
        # Calculate the error
        error = cost_function(expectedOutput, outputActivated)
        newError = pd.DataFrame({'epoch': [count], 'error': [error]})
        errorDF = pd.concat([errorDF, newError], ignore_index=True)
        print(f'Epoch: {count}, Error: {error}')

    actualDF = dataBase[['DATE', 'Skelton']]
    predictionDF = pd.DataFrame(predictions)

    return actualDF, predictionDF, errorDF

hiddenList, outputList = init_structure(perceptronStructure)
actualDF, predictionDF, errorDF = train_network(hiddenList, outputList)

with PdfPages('plots/error_vs_epochs.pdf') as pdf:
    line_plot(errorDF, pdf)
    scatter_plot([actualDF, predictionDF], 'Skelton', pdf, ['b', 'r'])