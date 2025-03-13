import numpy as np
import pandas as pd
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot, correlation_plot

perceptronStructure = (7, [(16, 'sigmoid'),(16, 'sigmoid'), (1, 'sigmoid')])
learningRate = 0.1
epochs = 1000

np.random.seed(42)

# Activation Functions
def get_activation(activation):
    '''Returns the required activation function'''

    # Sigmoid Function
    if activation == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))  
    
def get_activation_derivative(activation):
    '''Returns the derivative of the activation function'''

    # Sigmoid Function
    if activation == 'sigmoid':
        return lambda x: (1 / (1 + np.exp(-x)) ) * (1 - (1 / (1 + np.exp(-x)) ))

# Cost Functions
def cost_function(expectedOutput, outputActivated):
    '''Calculates the error of the neural network'''
    return ((expectedOutput - outputActivated) ** 2)

  
def cost_function_derivative(expectedOutput, outputActivated):
    '''Calculates the error of the neural network for the backwards pass'''
    return (expectedOutput - outputActivated)

# Normalization function
def min_max_scaler(data):
    '''Function to normalize the data'''
    return (data - data.min()) / (data.max() - data.min())

def min_max_reverser(data, dataMax, dataMin):
    '''Function to normalize the data'''
    return data * (dataMax - dataMin) + dataMin

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
    biasMatrix = np.random.uniform(-1, 1, (1, nodeDimensions[0][0]))

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

    return weightList, biasList

def forward_pass(inputData, weightList, biasList, layerStructure):
    '''Forward pass of the neural network'''

    # Create a list for the activated nodes and summed nodes
    summedList = []
    activatedList = []


    # Calculate for the hidden layers
    for layer, structure in enumerate(layerStructure):
        # Get activation function
        activationFunction = get_activation(structure[1])

        # Calculate the sum of the matrix
        sumMatrix = (inputData @ weightList[layer]) + biasList[layer]

        activatedMatrix = activationFunction(sumMatrix)

        # Append to the list
        summedList.append(sumMatrix)
        activatedList.append(activatedMatrix)

        # Update the input data
        inputData = activatedMatrix

    return activatedList, summedList


def backward_pass(expectedOutput, weightList, biasList, summedList, activatedList, inputData, layerStructure):
    '''Backward pass of the neural network'''

    # Cost of the network for that pass
    structureCost = cost_function_derivative(expectedOutput, activatedList[-1])

    # Calculate the delta values for the output layer
    activation_derivative = get_activation_derivative(layerStructure[-1][1])
    outputDelta = structureCost * activation_derivative(summedList[-1])

    # Calculate the delta values for the hidden layers
    deltaList = []
    deltaList.append(outputDelta)

    # Calculate the delta values for the hidden layers
    for layer in range(2, len(weightList)+1, 1):
        activation_derivative = get_activation_derivative(layerStructure[-layer][1])
        delta = (deltaList[-1] @ weightList[-(layer)+1].T) * activation_derivative(summedList[-layer])
        deltaList.append(delta)

    # Reverse the delta list
    deltaList.reverse()

    # Calculate the weights and biases with input layer as previous layer
    weightList[0] += (inputData.T @ deltaList[0]) * learningRate
    biasList[0] += deltaList[0] * learningRate
    

    # Update the weights and biases for the output layer
    for layer in range(1, len(weightList), 1):
        weightList[layer] += (activatedList[layer-1].T @ deltaList[layer]) * learningRate
        biasList[layer] += deltaList[layer] * learningRate

    return (weightList, biasList)


def train_network(trainingData, evaluationData, weightList, biasList, perceptronStructure, epochs):
    '''Trains the neural network with the training data'''

    count = 0

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
            activatedList, summedList = forward_pass(inputData, weightList, biasList, perceptronStructure[1])

            # Calculate the error and store it
            error = cost_function(expectedOutput, activatedList[-1])
            newMeanErrorDF = pd.DataFrame({'error': [error]})
            meanErrorDF = pd.concat([meanErrorDF, newMeanErrorDF], ignore_index=True)

            # Backward pass
            weightList, biasList = backward_pass(expectedOutput, weightList, biasList, summedList, activatedList, inputData.reshape(1, -1), perceptronStructure[1])

        count += 1
        # Store the mean error for the epoch
        newError = pd.DataFrame({'epoch': [count], 'error': [meanErrorDF['error'].mean()]})
        errorDF = pd.concat([errorDF, newError], ignore_index=True)
        if count % 100 == 0:
            print(f'Epoch: {count}, Error: {error}')


    print('Training complete')
    # Predict the data for the next year
    predictions = []
    for i, day in enumerate(evaluationData.itertuples(index=False), start=0):
        inputData = np.array([float(day[2]), float(day[3]), float(day[4]), float(day[5]), float(day[6]), float(day[7]), float(day[8])])

        # Forward pass
        activatedList, summedList = forward_pass(inputData, weightList, biasList, perceptronStructure[1])

        predictions.append({'DATE': day[0], 'Skelton': activatedList[-1]})
    # Create a database for the predictions
    predictionDF = pd.DataFrame(predictions)
        
    return predictionDF, errorDF

# Initialize the structure of the perceptron
weightList, biasList = init_structure(perceptronStructure)


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
trainingData.iloc[:, 1:] = min_max_scaler(trainingData.iloc[:, 1:])


sigmoid = get_activation('sigmoid')
sigmoid_derivative = get_activation_derivative('sigmoid')

# Train the neural network
predictionDF, errorDF  = train_network(trainingData[trainingData['DATE'].dt.year.isin([1993, 1994])],trainingData[trainingData['DATE'].dt.year == 1995],  weightList, biasList, perceptronStructure, epochs)

# Denormalize the data
predictionDF['Skelton'] = min_max_reverser(predictionDF['Skelton'], skeltonMax, skeltonMin)
trainingData['Skelton'] = min_max_reverser(trainingData['Skelton'], skeltonMax, skeltonMin)

# Plot the data
with PdfPages('plots/error_vs_epochs.pdf') as pdf:
    line_plot(errorDF, pdf)
    scatter_plot([trainingData[trainingData['DATE'].dt.year == 1995], predictionDF], 'Skelton', pdf, ['b', 'r'])
    correlation_plot([trainingData[trainingData['DATE'].dt.year == 1995], predictionDF], 'Skelton', 'Skelton', pdf, ['b', 'r'])