import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot
import reading_and_writing as rw
from scipy.special import expit
import matplotlib.pyplot as plt

perceptronStructure = (3,[(5,'sigmoid'), (1, 'sigmoid')])
learningRate = 0.2
epochs = 100000

# testing with sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def init_structure(perceptronStructure):
    '''Initializes the perceptron structure with random weights'''

    # Create lists to store the matrices
    weightsList = []
    deltaList = []
    biasList = []

    # Calculates the first level of the perceptron
    inputWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[0], perceptronStructure[1][0][0]))
    weightsList.append(inputWeightMatrix)

    deltaWeightMatrix = np.zeros((1, perceptronStructure[1][0][0]))
    deltaList.append(deltaWeightMatrix)

    inputBiasMatrix = np.random.uniform(-0.5, 0.5, (1, perceptronStructure[1][0][0]))
    biasList.append(inputBiasMatrix)

    # Calculates the rest of the levels of the perceptron
    for i in range(1, len(perceptronStructure[1])):
        hiddenWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[1][i-1][0], perceptronStructure[1][i][0]))
        weightsList.append(hiddenWeightMatrix)

        deltaWeightMatrix = np.zeros((1, perceptronStructure[1][i][0]))
        deltaList.append(deltaWeightMatrix)


        hiddenBiasMatrix = np.random.uniform(-0.5, 0.5, (1, perceptronStructure[1][i][0]))
        biasList.append(hiddenBiasMatrix)

    return (weightsList, deltaList, biasList)




def forward_pass(data, weightsAndBiases, perceptronStructure):
    '''Forward pass of the neural network'''

    # Initialize the node values matrix
    activatedValues = []
    summedValuesList = []
    # Prepare the matrices for calculation
    weightsList, deltaList, biasList = weightsAndBiases

    # Calculate values at the first hidden layer
    # activation = get_activation_function(perceptronStructure[1][0][1])
    summedValue  = np.dot(data, weightsList[0]) + biasList[0]
    previousLayer = sigmoid(summedValue)
    activatedValues.append(previousLayer)
    summedValuesList.append(summedValue)

    # Calculate values at the rest of the hidden layers
    for i in range(1, len(weightsList)):
        # Get this layers activation function
        # activation = get_activation_function(perceptronStructure[1][i][1])
        # Define the current hidden layer with matrix calculations
        summedValue  = np.dot(previousLayer, weightsList[i]) + biasList[i]
        previousLayer = sigmoid(np.dot(previousLayer, weightsList[i]) + biasList[i])
        # Append the current hidden layer to the node values matrix
        activatedValues.append(previousLayer)
        summedValuesList.append(summedValue)

    return activatedValues, summedValuesList


def backward_pass(weightsAndBiases, learningRate):
    '''Backward pass of the neural network'''

    # Get the weights and biases
    weightsList, deltaList, biasList = weightsAndBiases

    # Update the weights and biases
    for i in range(len(weightsList)):
        weightsList[i] += learningRate * activatedValues[i] * deltaList[i]
        biasList[i] += learningRate * np.sum(deltaList[i], axis=0, keepdims=True)

    return weightsList ,deltaList, biasList


def delta_function(weightsAndBiases, activatedValues, summedValues, dataOutput, perceptronStructure):
    '''Calculate the delta function'''

    # Get delta and weight matrices
    weightsList, deltaList, biasList = weightsAndBiases

    # define initial delta
    deltaInitial = error_function(dataOutput, activatedValues[-1]) * sigmoid_derivative(summedValues[-1])

    # Inset the initial delta into the delta list
    deltaList[-1] = deltaInitial
    
    # Calculate the rest of the deltas
    for i in range(len(activatedValues) - 2, -1, -1):
        # Get the activation function
        # activation = get_activation_derivative(perceptronStructure[1][i][1])

        # Calculate the delta
        delta = np.dot(deltaList[i+1], weightsList[i+1].T) * sigmoid_derivative(summedValues[i])

        # Insert the delta into the delta list
        deltaList[i] = delta

    return weightsList, deltaList, biasList



'''-----------------------------Error Functions------------------------------------'''
def error_function(dataOutput, activatedValues):
    '''Calculate the error function'''
    #error = 0.5 * np.sum((dataOutput - activatedValues[-1]) ** 2)
    error = np.sum((dataOutput - activatedValues[-1]))
    return error


def min_max_scaler(data):
    '''Function to normalize the data'''
    return (data - data.min()) / (data.max() - data.min())


'''-----------------------------Run Time------------------------------------'''
# Load the data
dataBase = rw.read_data_all('dataSet.db', 'data_table')
dataBase.iloc[:, 1:] = min_max_scaler(dataBase.iloc[:, 1:])

# Initialize an empty DataFrame to store errors
errorDF = pd.DataFrame(columns=['epoch', 'error'])

# Initialize the weights and biases
weightsAndBiases = init_structure(perceptronStructure)

for epoch in range(epochs):
    predictions = []
    for i, day in enumerate(dataBase.itertuples(index=False), start=0):
        # No errors for our of range datapoints (first and last)
        if i == len(dataBase)-1:
            continue

        # Get the previous and next day
        nextDay = tuple(dataBase.iloc[i + 1])

        inputData = np.array([float(day[2]), float(day[8]), float(day[4])])
        outputData = np.array(float(nextDay[4]))

        # Forward pass
        activatedValues, summedValues = forward_pass(inputData, weightsAndBiases, perceptronStructure)

        # Backward pass
        weightsAndBiases = delta_function(weightsAndBiases, activatedValues, summedValues, outputData, perceptronStructure)

        # Update the weights and biases
        weightsAndBiases = backward_pass(weightsAndBiases, learningRate)

        predictions.append({'DATE': nextDay[0], 'Skelton': activatedValues[-1][0]})

    # Calculate the error
    error = error_function(outputData, activatedValues)
    newError = pd.DataFrame({'epoch': [epoch], 'error': [error]})
    errorDF = pd.concat([errorDF, newError], ignore_index=True)
    print(f'Epoch: {epoch}, Error: {error}')

actualDF = dataBase[['DATE', 'Skelton']]

# Create a DataFrame for the predictions
predictionDF = pd.DataFrame(predictions)

with PdfPages('plots/error_vs_epochs.pdf') as pdf:
    line_plot(errorDF, pdf)
    scatter_plot([actualDF, predictionDF], 'Skelton', pdf, ['b', 'r'])