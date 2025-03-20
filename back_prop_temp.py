import numpy as np
import pandas as pd
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot, correlation_plot, plot_text

import time


class MLP:

    def __init__(self, dataSet:pd.DataFrame, nodeStructure:list[tuple], learningRate:float, trainingEpochs:int, PredictedColumn:str):
        '''Initialises the perceptron structure'''

        # Determine the number of input dimensions
        self.inputDimensions = (dataSet.shape[1]-2)

        # Normalise the training data
        self.dataFrameMin = dataSet[PredictedColumn].min()
        self.dataFrameMax = dataSet[PredictedColumn].max()
        self.dataSet = dataSet.copy()
        self.dataSet.iloc[:, 1:] = self.min_max_scaler(self.dataSet.iloc[:, 1:])

        # Split the data into training and testing
        self.trainingData = self.dataSet[dataSet['DATE'].dt.year.isin([1993, 1994])]
        self.testingData = self.dataSet[dataSet['DATE'].dt.year == 1995]


        # Store the structure of the network
        self.nodeStructure = nodeStructure
        self.learningRate = learningRate
        self.trainingEpochs = trainingEpochs

        # Create lists for weights and biases
        self.weightList = []
        self.biasList = []

        # Populate the structure lists
        previousLayer = self.inputDimensions

        for layer in self.nodeStructure:
            # Create matrices for the weights and biases
            weightMatrix = np.random.uniform(-1, 1, (previousLayer, layer[0]))
            biasMatrix = np.random.uniform(-1, 1, (1, layer[0]))

            # Append to the lists
            self.weightList.append(weightMatrix)
            self.biasList.append(biasMatrix)

            # Update the previous layer
            previousLayer = layer[0]

        # Create dataframe for storing the error per epoch
        self.epochErrorDF = pd.DataFrame(columns=['Epoch', 'Error'])

        # Create dataframe for storing the predictions
        self.predictionDF = pd.DataFrame(columns=['DATE', 'Skelton'])

    def min_max_scaler(self, dataFrame):
        '''Function to normalise the data'''
        return (dataFrame - self.dataFrameMin) / (self.dataFrameMax - self.dataFrameMin)

    def min_max_reverser(self, dataFrame):
        '''Function to normalize the data'''
        return dataFrame * (self.dataFrameMax - self.dataFrameMin) + self.dataFrameMin




    def train_network(self) -> pd.DataFrame:
        '''Function to train the neural network'''

        # Iterate through the training epochs
        for epoch in range(self.trainingEpochs):

            # Create a dataframe to store the mean error for the epoch
            meanErrorList = []

            # Convert data to numpy array
            inputDataArray = self.trainingData.iloc[:, 2:].values
            expectedOutputArray = self.trainingData.iloc[:, 1].values

            # Iterate though each day in the training data
            for i in range(len(inputDataArray)):

                # Split the data point into input and output
                inputData, expectedOutput = inputDataArray[i], expectedOutputArray[i]

                # Pass the data through the network
                activatedList, summedList = self.forward_pass(inputData)

                meanErrorList.append(self.error_storing(expectedOutput, activatedList[-1]))

                # Backward pass through the network
                self.backward_pass(expectedOutput, summedList, activatedList, inputData)

            # Convert errors into dataframe
            meanErrorDF = pd.DataFrame({'Error' : meanErrorList})

            # Calculate the mean error for the epoch
            meanError = meanErrorDF['Error'].mean()

            # Store the mean error for the epoch
            newError = pd.DataFrame({'Epoch': [epoch], 'Error': [meanError]})
            self.epochErrorDF = pd.concat([self.epochErrorDF, newError], ignore_index=True)

            # Print the error for the epoch
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Error: {meanError}')



    def cost_function(self, expectedOutput:np.array, predictedOutput:np.array) -> float:
        '''Function to calculate the error of the network'''

        # Calculate the error
        return ((expectedOutput - predictedOutput) ** 2)

    def cost_function_derivative(self, expectedOutput:np.array, predictedOutput:np.array) -> float:
        '''Function to calculate the derivative of the error function'''

        # Calculate the derivative of the error
        return (expectedOutput - predictedOutput)
    
    def error_storing(self, expectedOutput:np.array, predictedOutput:np.array) -> pd.DataFrame:
        '''Function to store the error of the network'''

        # De Scale the data
        deScaledExpected = self.min_max_reverser(expectedOutput)
        deScaledPredicted = self.min_max_reverser(predictedOutput)

        # Calculate the error of the network
        networkError = self.cost_function(deScaledExpected, deScaledPredicted)

        return networkError




    def forward_pass(self, inputData:np.array) -> tuple[np.array, np.array]:
        '''Function to pass data through the network'''

        # Create a list for the activated nodes and summed nodes
        summedList = []
        activatedList = []

        # Calculate each layers output
        for layer, structure in enumerate(self.nodeStructure):
            # Get activation function for the layer
            activationFunction = self.get_activation(structure[1])

            # Calculate the sum of the matrix
            sumMatrix = (inputData @ self.weightList[layer]) + self.biasList[layer]

            # Calculate the activated matrix
            activatedMatrix = activationFunction(sumMatrix)

            # Append to the lists
            summedList.append(sumMatrix)
            activatedList.append(activatedMatrix)

            # Update the input data for the next layer
            inputData = activatedMatrix

        return activatedList, summedList

    def get_activation(self, activation:str) -> callable:
        '''Function to return the activation function'''

        # Sigmoid Function
        if activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))  
        

    def backward_pass(self, expectedOutput:np.array, summedList:np.array, activatedList:np.array, inputData:np.array):
        '''Function to pass data through the network'''

        # Calculate the delta values for the network
        deltaList = self.delta_calculator(expectedOutput, activatedList, summedList)

        #  Set previous output as input values
        previousOutput = inputData.reshape(1, -1)

        # Iterate through the layers of the network
        for layer in range(len(self.nodeStructure)):

            # Update the weights and biases for the layer
            self.weightList[layer] += (previousOutput.T @ deltaList[layer]) * self.learningRate
            self.biasList[layer] += deltaList[layer] * self.learningRate

            # Update the previous output
            previousOutput = activatedList[layer]


    def delta_calculator(self, expectedOutput:np.array, activatedList:np.array, summedList:np.array) -> np.array:

        # Calculate the cost of the network for that pass
        structureCost = self.cost_function_derivative(expectedOutput, activatedList[-1])

        # Calculate the delta values for the output layer
        activation_derivative = self.get_activation_derivative(self.nodeStructure[-1][1])
        outputDelta = structureCost * activation_derivative(summedList[-1])

        # Calculate the delta values for the hidden layers
        deltaList = []
        deltaList.append(outputDelta)

        # Calculate the delta values for the hidden layers
        for layer in range(2, len(self.nodeStructure)+1, 1):

            # Get the activation derivative for the layer
            activation_derivative = self.get_activation_derivative(self.nodeStructure[-layer][1])

            # Calculate the delta for the layer
            delta = (deltaList[-1] @ self.weightList[-(layer)+1].T) * activation_derivative(summedList[-layer])

            # Append to the list
            deltaList.append(delta)

        # Reverse the delta list to match order of the rest of the structure lists
        deltaList.reverse()

        return deltaList
        
    def get_activation_derivative(self, activation:str) -> callable:
        '''Function to return the derivative of the activation function'''

        # Sigmoid Function
        if activation == 'sigmoid':
            return lambda x: (1 / (1 + np.exp(-x)) ) * (1 - (1 / (1 + np.exp(-x)) ))
        

    def predict(self):
        '''Function to predict the output of the network'''

        # Create a list to store the predictions
        predictionList = []

        # Create a list to store the error
        errorList = []

        # Convert data to numpy array
        inputDataArray = self.testingData.iloc[:, 2:].values
        dateArray = self.testingData.iloc[:, 0].values
        expectedOutputArray = self.testingData.iloc[:, 1].values

        # Iterate though each day in the training data
        for i in range(len(inputDataArray)):

            # Split the data point into input and output
            inputData, date, expectedOutput = inputDataArray[i], dateArray[i], expectedOutputArray[i]

            # Pass the data through the network
            activatedList, summedList = self.forward_pass(inputData)

            # De Scale the data
            deScaledPredicted = self.min_max_reverser(activatedList[-1])

            # Error calculation
            passError = expectedOutput - deScaledPredicted

            # Append to the list
            errorList.append(passError)

            # Append the prediction to the list
            predictionList.append({'DATE': date, 'Skelton': deScaledPredicted})

        # Create a dataframe for the predictions
        self.predictionDF = pd.DataFrame(predictionList)

        # Create a dataframe for the errors
        self.predictErrorDF = pd.DataFrame({'Error' : errorList})

        


def create_data(predictedColumn:str) -> pd.DataFrame:
    '''Function to create a sample dataset'''

    # Load the data
    dataBase = rw.read_data_all('dataSet.db', 'data_table')

    # create database for the predictors for the data
    trainingData = pd.DataFrame()
    trainingData['DATE'] = dataBase['DATE']

    # Predictand column
    trainingData[predictedColumn] = dataBase['Skelton'].shift(-1)

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

    return trainingData

        
def main():
    '''Main function to run the program'''

    # Name Predictand Column
    predictedColumn = 'Predicted Skelton'

    # Load the data
    dataSet = create_data(predictedColumn)

    trainingEpochs = 3000

    for learningRate in np.arange(0.05, 0.15, 0.02):
        for i in range(6, 18, 2):

            # Define the structure of the network
            nodeStructure = [(i, 'sigmoid'), (1, 'sigmoid')]

            # Create the neural network
            neuralNetwork = MLP(dataSet, nodeStructure, learningRate, trainingEpochs, predictedColumn)

            # Train the neural network
            neuralNetwork.train_network()

            # Predict the output of the network
            neuralNetwork.predict()

            # Plot the data
            with PdfPages(f'plots/testing/{learningRate}LR_{i}Nodes.pdf') as pdf:
                line_plot(neuralNetwork.epochErrorDF, pdf)
                scatter_plot([dataSet[dataSet['DATE'].dt.year == 1995], neuralNetwork.predictionDF], 'Skelton', pdf, ['b', 'r'])
                correlation_plot([dataSet[dataSet['DATE'].dt.year == 1995], neuralNetwork.predictionDF], predictedColumn, 'Skelton', pdf, ['b', 'r'])
                plot_text(f'Learning Rate: {learningRate}, Nodes: {i}, Error: {neuralNetwork.predictErrorDF['Error'].mean()}', pdf)

start_time = time.time()

if __name__ == '__main__':
    main()

end_time = time.time()
print(f'Time taken: {end_time - start_time} seconds')

# 1000 epochs, 3 layers (7, 16,16,1), LR = 0.1, time = 546 seconds