# back Propogation algorithm
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot
import reading_and_writing as rw
from scipy.special import expit
import matplotlib.pyplot as plt

perceptronStructure = (4,[(6,'sigmoid'), (1, 'sigmoid')])
learningRate = 0.05
epochs = 50


#re do this bit for sure

def get_activation_function(name):
    """
    Returns an activation function that works with both np.matrix and np.ndarray.
    """
    activations = {
        "relu": lambda x: np.maximum(0, np.asarray(x)),  
        "sigmoid": expit,  
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
        "sigmoid": lambda x: (s := expit(x)) * (1 - s),  # Sigmoid derivative: s * (1 - s)
        "tanh": lambda x: 1 - np.tanh(x) ** 2,  # Derivative of tanh: 1 - tanh^2(x)
        "linear": lambda x: np.ones_like(x),  # Derivative of linear (identity) function is 1
        "softmax": lambda x: (s := np.exp(x) / np.sum(np.exp(x))) * (1 - s)  # Softmax derivative (simplified)
    }

    return derivatives.get(name, lambda x: np.ones_like(x))  # Default to 1 if not found


# testing with sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def init_structure(perceptronStructure):
    '''Initializes the perceptron structure with random weights'''

    weightsList = []
    biasList = []

    inputWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[0], perceptronStructure[1][0][0]))
    weightsList.append(inputWeightMatrix)

    inputBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[1][0][0]))
    biasList.append(inputBiasMatrix)

    for i in range(1, len(perceptronStructure[1])):
        hiddenWeightMatrix = np.random.uniform(-1, 1, (perceptronStructure[1][i-1][0], perceptronStructure[1][i][0]))
        weightsList.append(hiddenWeightMatrix)


        hiddenBiasMatrix = np.random.uniform(-1, 1, (1, perceptronStructure[1][i][0]))
        biasList.append(hiddenBiasMatrix)

    return (weightsList, weightsList, biasList)


    

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
        activation = get_activation_function(perceptronStructure[1][i][1])
        # Define the current hidden layer with matrix calculations
        summedValue  = np.dot(previousLayer, weightsList[i]) + biasList[i]
        previousLayer = sigmoid(np.dot(previousLayer, weightsList[i]) + biasList[i])
        # Append the current hidden layer to the node values matrix
        activatedValues.append(previousLayer)
        summedValuesList.append(summedValue)

    return activatedValues, summedValuesList

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

def backward_pass(weightsAndBiases, activatedValues, summedValues, dataOutput, perceptronStructure):
    weightsList, deltaList, biasList = delta_function(weightsAndBiases, activatedValues, summedValues, dataOutput, perceptronStructure)

    # Update the weights and biases
    for i in range(0, len(weightsList)):
        weightsList[i] += learningRate * np.dot(activatedValues[i].T, deltaList[i])
        biasList[i] += learningRate * np.sum(deltaList[i], axis=0, keepdims=True)

def error_function(dataOutput, activatedValues):
    '''Calculate the error function'''
    # error = 0.5 * np.sum((dataOutput - activatedValues[-1]) ** 2)
    error = np.sum((dataOutput - activatedValues[-1]))
    return error

def min_max_scaler(data):
    '''Function to normalize the data'''
    return (data - data.min()) / (data.max() - data.min())


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
        if (i == 0 or i == len(dataBase)-1):
            continue

        # Get the previous and next day
        prevDay = tuple(dataBase.iloc[i - 1])
        nextDay = tuple(dataBase.iloc[i + 1])
        '''
        inputData = np.array([float(day[2]), float(day[4]), float(prevDay[8]), float(prevDay[5])])
        outputData = np.array(nextDay[4]) '''
        inputData = dataBase.iloc[i - 1, [2, 4, 8, 5]].values.astype(float)
        outputData = np.array([dataBase.iloc[i + 1, 4]], dtype=float)

        # forward pass
        activatedValues, summedValues = forward_pass(inputData, weightsAndBiases, perceptronStructure)
        print(weightsAndBiases[0])
        print('-----------------')
        weightsList, deltaList, biasList = delta_function(weightsAndBiases, activatedValues, summedValues, outputData, perceptronStructure)
        print(weightsList)
        print('-----------------')
        print(deltaList)
        print('-----------------')
        print(biasList)



        input('Press enter to continue')

        # backward pass
        #weightsAndBiases = backward_pass(activatedValues, summedValues, weightsAndBiases, outputData, learningRate, perceptronStructure)

        predictions.append({'DATE': nextDay[0], 'Skelton': activatedValues[-1][0]})

    # calculate the error
    error = error_function(outputData, activatedValues)

    # Append the error to the DataFrame
    newError = pd.DataFrame({'epoch': [epoch], 'error': [error]})
    errorDF = pd.concat([errorDF, newError], ignore_index=True)

    print(f'Epoch: {epoch}, Error: {error}')

actualDF = dataBase[['DATE', 'Skelton']]

# Create a DataFrame for the predictions
predictionDF = pd.DataFrame(predictions)

with PdfPages('plots/error_vs_epochs.pdf') as pdf:
    line_plot(errorDF, pdf)
    scatter_plot([actualDF, predictionDF], 'Skelton', pdf, ['b', 'r'])




