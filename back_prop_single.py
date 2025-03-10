import numpy as np
import pandas as pd
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot, correlation_plot


perceptronStructure = (7, 16, 1)
learningRate = 0.2
epochs = 10000


# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def cost_function(expectedOutput, outputActivated):
    '''Calculates the error of the neural network'''
    return ((expectedOutput - outputActivated) ** 2)

  
def cost_function_derivative(expectedOutput, outputActivated, dfLength):
    '''Calculates the error of the neural network for the backwards pass'''
    return (expectedOutput - outputActivated) * (2/dfLength)

# Normalization function
def min_max_scaler(data):
    '''Function to normalize the data'''
    return (data - data.min()) / (data.max() - data.min())

def min_max_reverser(data, dataMax, dataMin):
    '''Function to normalize the data'''
    return data * (dataMax - dataMin) + dataMin


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
    hiddenSum = (inputData @ hiddenList[0]) + hiddenList[1]
    hiddenActivated = sigmoid(hiddenSum)

    # Calculate for the output layer
    outputSum = (hiddenActivated @ outputList[0]) + outputList[1]
    outputActivated = sigmoid(outputSum)

    return hiddenSum, hiddenActivated, outputSum, outputActivated


def backward_pass(expectedOutput, hiddenSum, hiddenActivated, outputSum, outputActivated, hiddenList, outputList, inputData, dfLength):
    '''Backward pass of the neural network'''

    # Calculate the cost of the perceptron structure
    structureCost = cost_function_derivative(expectedOutput, outputActivated, dfLength)

    # Calculate the delta values for the output layer
    outputDelta = structureCost * sigmoid_derivative(outputSum)

    # Calculate the delta values for the hidden layer
    hiddenDelta = (outputDelta @ outputList[0].T) * sigmoid_derivative(hiddenSum)


    # Update the weights and biases for the output layer
    outputList[0] += (hiddenActivated.T @ outputDelta) * learningRate
    outputList[1] += outputDelta * learningRate

    # Update the weights and biases for the hidden layer
    hiddenList[0] += (inputData.T @ hiddenDelta) * learningRate
    hiddenList[1] += hiddenDelta * learningRate

    return (hiddenList, outputList)


def train_network(trainingData, evaluationData, hiddenList, outputList):
    '''Trains the neural network with the training data'''
    count = 0
    dfLength = len(trainingData)

    # Initialize an empty DataFrame to store errors
    errorDF = pd.DataFrame(columns=['epoch', 'error'])

    while count < epochs:
        # Initialize an empty DataFrame to store the mean error for the epoch
        meanErrorDF = pd.DataFrame(columns=['error'])

        # Iterate through the training data
        for i, day in enumerate(trainingData.itertuples(index=False), start=0):

            inputData = np.array([float(day[2]), float(day[3]), float(day[4]), float(day[5]), float(day[6]), float(day[7]), float(day[8])])
            expectedOutput = np.array([float(day[1])])

            # Forward pass
            hiddenSum, hiddenActivated, outputSum, outputActivated = forward_pass(inputData, hiddenList, outputList)

            # Calculate the error and store it
            error = cost_function(expectedOutput, outputActivated)
            newMeanErrorDF = pd.DataFrame({'error': [error]})
            meanErrorDF = pd.concat([meanErrorDF, newMeanErrorDF], ignore_index=True)

            # Backward pass
            hiddenList, outputList = backward_pass(expectedOutput, hiddenSum, hiddenActivated, outputSum, outputActivated, hiddenList, outputList, inputData.reshape(1, -1), dfLength)

        count += 1
        # Store the mean error for the epoch
        newError = pd.DataFrame({'epoch': [count], 'error': [meanErrorDF['error'].mean()]})
        errorDF = pd.concat([errorDF, newError], ignore_index=True)
        if count % 50 == 0:
            print(f'Epoch: {count}, Error: {error}')

    print('Training complete')
    # Predict the data for the next year
    predictions = []
    for i, day in enumerate(evaluationData.itertuples(index=False), start=0):
        inputData = np.array([float(day[2]), float(day[3]), float(day[4]), float(day[5]), float(day[6]), float(day[7]), float(day[8])])

        # Forward pass
        hiddenSum, hiddenActivated, outputSum, outputActivated = forward_pass(inputData, hiddenList, outputList)

        predictions.append({'DATE': day[0], 'Skelton': outputActivated[0]})
    # Create a database for the predictions
    predictionDF = pd.DataFrame(predictions)
        
    return predictionDF, errorDF

# Initialize the structure of the perceptron
hiddenList, outputList = init_structure(perceptronStructure)


# Load the data
dataBase = rw.read_data_all('dataSet.db', 'data_table')

# create database for the predictors for the data
trainingData = pd.DataFrame()
trainingData['DATE'] = dataBase['DATE']

# Predictand column
trainingData['Predicted Skelton'] = dataBase['Skelton'].shift(-1)

# Predictors
trainingData['Skelton'] = dataBase['Skelton']
trainingData['Westwick'] = dataBase['Westwick']
trainingData['Flow Average'] = (dataBase['Skelton'] + dataBase['Westwick'] + dataBase['Skip Bridge'] + dataBase['Crakehill']) / 4
trainingData['Rainfall Average'] = (dataBase['Arkengarthdale'] + dataBase['East Cowton'] + dataBase['Malham Tarn'] + dataBase['Snaizeholme']) / 4
trainingData['3 Day Rain'] = trainingData['Rainfall Average'] + trainingData['Rainfall Average'].shift(1) + trainingData['Rainfall Average'].shift(2)
trainingData['Average Flow + Average Rain'] = trainingData['Rainfall Average'] + trainingData['Flow Average']
trainingData['Skelton + Average Rain'] = trainingData['Skelton'] + trainingData['Rainfall Average']

# Drop na values from shifting
trainingData = trainingData.dropna()

# Store max Skelton values
skeltonMax = trainingData['Skelton'].max()
skeltonMin = trainingData['Skelton'].min()

# Normalize the data
trainingData.iloc[:, 1:]= min_max_scaler(trainingData.iloc[:, 1:])

# Train the neural network
predictionDF, errorDF  = train_network(trainingData[trainingData['DATE'].dt.year.isin([1993, 1994])],trainingData[trainingData['DATE'].dt.year == 1995],  hiddenList, outputList)

# Denormalize the data
predictionDF['Skelton'] = min_max_reverser(predictionDF['Skelton'], skeltonMax, skeltonMin)
trainingData['Skelton'] = min_max_reverser(trainingData['Skelton'], skeltonMax, skeltonMin)

# Plot the data
with PdfPages('plots/error_vs_epochs.pdf') as pdf:
    line_plot(errorDF, pdf)
    scatter_plot([trainingData[trainingData['DATE'].dt.year == 1995], predictionDF], 'Skelton', pdf, ['b', 'r'])
    correlation_plot([trainingData[trainingData['DATE'].dt.year == 1995], predictionDF], 'Skelton', 'Skelton', pdf, ['b', 'r'])