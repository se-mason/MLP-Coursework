import numpy as np
import pandas as pd
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_plotting import line_plot, scatter_plot, correlation_plot, line_plot_multi
import matplotlib.pyplot as plt

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

        # Reshape into a 3D array
        self.dataSet = self.dataSet.to_numpy().reshape(dataSet.shape[0], 1, dataSet.shape[1])

        # Split the data into training and testing
        self.trainingData = self.dataSet[dataSet['DATE'].dt.year.isin([1993, 1994])]
        self.testingData = self.dataSet[dataSet['DATE'].dt.year == 1995]
        self.evaluationData = self.dataSet[dataSet['DATE'].dt.year == 1996]

        # Define batch size start point
        self.batchSize = 1

        # Define the momentum constant
        self.momentumConstant = 0.7

        # Create Bold Driver Learning Rate
        self.boldDriverLearningRate = learningRate
        self.boldDriverCap = 20
        self.boldDriverIncrease = 1.3

        # Store the structure of the network
        self.nodeStructure = nodeStructure
        self.learningRate = learningRate
        self.trainingEpochs = trainingEpochs

        # Create lists for weights and biases
        self.weightList = []
        self.biasList = []
        self.previousWeightList = []

        # Populate the structure lists
        previousLayer = self.inputDimensions

        for layer in self.nodeStructure:
            # Create matrices for the weights and biases
            weightMatrix = np.random.uniform(-1, 1, (previousLayer, layer[0]))
            biasMatrix = np.random.uniform(-0.01, 0.01, (1, layer[0]))
            previousWeightMatrix = np.zeros((previousLayer, layer[0]))

            # Append to the lists
            self.weightList.append(weightMatrix)
            self.biasList.append(biasMatrix)
            self.previousWeightList.append(previousWeightMatrix)

            # Update the previous layer
            previousLayer = layer[0]

        # Create dataframe for storing the error per epoch
        self.epochErrorDF = pd.DataFrame(columns=['Epoch', 'Error'])

        # Create dataframe for storing the predictions
        self.predictionDF = pd.DataFrame(columns=['DATE', 'Skelton'])

        # Create dataframe for storing the predictions of the training data
        self.predictionDFTesting = pd.DataFrame(columns=['DATE', 'Skelton'])

        # Create dataframe for storing the error per epoch
        self.epochErrorDFTesting = pd.DataFrame(columns=['Epoch', 'Error'])

        # Create dataframe for storing the predictions
        self.evaluationDF = pd.DataFrame(columns=['DATE', 'Skelton'])
        self.entirePredictDF = pd.DataFrame(columns=['DATE', 'Skelton'])


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

            # Define the batch data
            batchData = np.array_split(self.trainingData, self.batchSize)

            # Define the mean error list
            meanErrorList = []

            # Iterate through the batches
            for batch in batchData:

                # Split the data point into input and output, maintaining the 3D structure
                inputData, expectedOutput = batch[:, :, 2:], batch[:, :, 1:2]

                # Pass the data through the network
                activatedList, summedList = self.forward_pass(inputData, self.weightList, self.biasList) 

                # Calculate the error of the network
                meanErrorbatch = self.error_storing(expectedOutput, activatedList[-1])
                
                # Reshape dimensions
                meanErrorbatch = meanErrorbatch.squeeze(axis=1)

                # Average the outputs and store in the list
                meanErrorList.append(meanErrorbatch.mean())

                # Backward pass through the network
                self.line_search(expectedOutput, activatedList, summedList, inputData)


            # Convert errors into dataframe
            meanErrorDF = pd.DataFrame({'Error' : meanErrorList})

            # Calculate the mean error for the epoch
            meanError = meanErrorDF['Error'].mean()

            # Store the mean error for the epoch
            newError = pd.DataFrame({'Epoch': [epoch], 'Error': [meanError]})
            self.epochErrorDF = pd.concat([self.epochErrorDF, newError], ignore_index=True)

            # Try against the testing data
            networkErrorTesting = self.predict_train()

            # Reshape dimensions
            networkErrorTesting = networkErrorTesting.squeeze(axis=1)

            # Store the mean error for the epoch
            networkErrorTesting = pd.DataFrame({'Epoch': [epoch], 'Error': [networkErrorTesting.mean()]})
            self.epochErrorDFTesting = pd.concat([self.epochErrorDFTesting, networkErrorTesting], ignore_index=True)

            # Only allow changes every 200 epochs to force the network to learn at certain points, and enforces at least 2 epochs
            if epoch % 200 == 0 and len(self.epochErrorDF) > 1:

                # Define the last 200 points for an accurate gradient
                recentErrors = self.epochErrorDF['Error'][-200:] if len(self.epochErrorDF) >= 200 else self.epochErrorDF['Error']

                # Calculate the error gradient with respect to the batch size
                errorGradient = np.gradient(recentErrors)

                print(f'Error Gradient: {errorGradient[-1]}')

                # Doesn't allow batch size to be greater than the training data
                if self.batchSize == 32:
                    None

                # Increase the batch size
                elif abs(errorGradient[-1]) < 0.09:
                    self.batchSize = self.batchSize * 2
                    self.momentumConstant = self.momentumConstant * 0.6

                    # Forces batch size to be the same as the training data
                    if self.batchSize > 32:
                        self.batchSize = 32

                # Update the learning rate
                if abs(errorGradient[-1]) < 0.07:
                    self.boldDriverLearningRate = self.boldDriverLearningRate * self.boldDriverIncrease

                    # Caps the learning rate 
                    if self.boldDriverLearningRate > (1+self.boldDriverCap)*self.learningRate:
                        self.boldDriverLearningRate = (1+self.boldDriverCap)*self.learningRate



            # Print the error for the epoch
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Error: {meanError}, Batch Size: {self.batchSize}')




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




    def forward_pass(self, inputData:np.array, weightList:np.array, biasList:np.array) -> tuple[np.array, np.array]:
        '''Function to pass data through the network'''

        # Create a list for the activated nodes and summed nodes
        summedList = []
        activatedList = []

        # Calculate each layers output
        for layer, structure in enumerate(self.nodeStructure):
            # Get activation function for the layer
            activationFunction = self.get_activation(structure[1])

            # Calculate the sum of the matrix
            sumMatrix = (inputData @ weightList[layer]) + biasList[layer]

            # Squeeze the sum matrix
            sumMatrix = np.array(sumMatrix.squeeze(axis=1), dtype=np.float64)

            # Calculate the activated matrix
            activatedMatrix = activationFunction(sumMatrix)

            # Reshape the activated matrix
            activatedMatrix = activatedMatrix[:, np.newaxis, :]
            sumMatrix = sumMatrix[:, np.newaxis, :]

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
        

    def backward_pass(self, activatedList:np.array, inputData:np.array, lineSearchConstant:float, deltaList:np.array):
        '''Function to pass data through the network'''

        # Create a list for the temporary weights and biases
        temporaryWeightList = []
        temporaryBiasList = []

        #  Set previous output as input values
        previousOutput = inputData

        # Define the step size for the line search
        stepSize = self.boldDriverLearningRate * lineSearchConstant

        # Iterate through the layers of the network
        for layer in range(len(self.nodeStructure)):

            # Update the weights and biases for the layer
            weightUpdate  = ((np.transpose(previousOutput, (0, 2, 1)) @ deltaList[layer]) * stepSize)
            biasUpdate = (deltaList[layer] * stepSize)

            # Update the previous output
            previousOutput = activatedList[layer]

            # Momentum update
            momentumValue = (self.weightList[layer] - self.previousWeightList[layer]) * self.momentumConstant

            # Update the weights and biases
            temporaryWeightList.append(self.weightList[layer] + (weightUpdate.mean(axis=0, dtype=np.float64) + momentumValue))
            temporaryBiasList.append(self.biasList[layer] + biasUpdate.mean (axis=0, dtype=np.float64))

        return temporaryWeightList, temporaryBiasList


    def line_search(self, expectedOutput:np.array, activatedList:np.array, summedList:np.array, inputData:np.array):
        '''Function to perform a line search and optimise the learning rate'''

        # Define the next two steps
        learningRateAlpha = self.learningRate * 20
        learningRateBeta = self.learningRate * 60

        # Define the initial error and delta list
        deltaList, structureCost = self.delta_calculator(expectedOutput, activatedList, summedList)

        # Calculate the weights for the first step

        # Calculate the forward pass for the first step
        activatedListAlpha, summedListAlpha = self.forward_pass(inputData, self.weightList, self.biasList)

        # Calculate the error for the first step
        structureCostAlpha = self.error_storing(expectedOutput, activatedListAlpha[-1])

        # Calculate the weights for the second step

        # Calculate the forward pass for the second step
        activatedListBeta, summedListBeta = self.forward_pass(inputData, self.weightList, self.biasList)

        # Calculate the error for the second step
        structureCostBeta = self.error_storing(expectedOutput, activatedListBeta[-1])

        # Use a quadratic model to approximate the minimum learning rate
        X = np.array([
            [0**2, 0, 1],
            [learningRateAlpha**2, learningRateAlpha, 1],
            [learningRateBeta**2, learningRateBeta, 1]
        ])
        
        
        Y = np.array([structureCost.mean(), structureCostAlpha.mean(), structureCostBeta.mean()])
        
        # Solve for quadratic coefficients [a, b, c]
        a, b, c = np.linalg.solve(X, Y)
        
        # Compute the minimum learning rate
        stepSize = -b / (2 * a)

        # Calculate the backwards pass for the chose step size
        newWeightList, newBiasList = self.backward_pass(activatedList, inputData, stepSize, deltaList)

        self.weightList = newWeightList
        self.biasList = newBiasList

        # Momentum update
        self.previousWeightList = newWeightList


    def delta_calculator(self, expectedOutput:np.array, activatedList:np.array, summedList:np.array) -> np.array:
        '''Function to calculate the delta values for the network'''

        # Squeeze the matrices
        activatedList[-1] = np.array(activatedList[-1].squeeze(axis=1), dtype=np.float64)
        summedList[-1] = np.array(summedList[-1].squeeze(axis=1), dtype=np.float64)
        expectedOutput = np.array(expectedOutput.squeeze(axis=1), dtype=np.float64)

        # Calculate the cost of the network for that pass
        structureCost = self.cost_function_derivative(expectedOutput, activatedList[-1])

        # Return the activated list to the correct shape
        # activatedList[-1] = activatedList[-1][:, np.newaxis, :]

        # Calculate the delta values for the output layer
        activation_derivative = self.get_activation_derivative(self.nodeStructure[-1][1])
        outputDelta = structureCost * activation_derivative(summedList[-1])

        # Store the output delta
        previousDelta = outputDelta

        # Reshape the output delta
        outputDelta = outputDelta[:, np.newaxis, :]

        # Calculate the delta values for the hidden layers
        deltaList = []
        deltaList.append(outputDelta)

        # Calculate the delta values for the hidden layers
        for layer in range(2, len(self.nodeStructure)+1, 1):

            # Get the activation derivative for the layer
            activation_derivative = self.get_activation_derivative(self.nodeStructure[-layer][1])

            # Squeeze the sum matrix
            summedList[-layer] = np.array(summedList[-layer].squeeze(axis=1), dtype=np.float64)

            # Calculate the delta for the layer
            delta = (previousDelta @ self.weightList[-(layer)+1].T) * activation_derivative(summedList[-layer])

            # Update the previous delta
            previousDelta = delta

            # Reshape the delta
            delta = delta[:, np.newaxis, :]

            # Append to the list
            deltaList.append(delta)


        # Reverse the delta list to match order of the rest of the structure lists
        deltaList.reverse()

        return deltaList, structureCost
        
    def get_activation_derivative(self, activation:str) -> callable:
        '''Function to return the derivative of the activation function'''

        # Sigmoid Function
        if activation == 'sigmoid':
            return lambda x: (1 / (1 + np.exp(-x)) ) * (1 - (1 / (1 + np.exp(-x)) ))
        

    def predict(self):
        '''Function to predict the output of the network'''

        # Split the data point into input and output, maintaining the 3D structure
        inputData, inputDate, expectedData = self.testingData[:, :, 2:],  self.testingData[:, :, 0:1], self.testingData[:, :, 1:2]

        # Pass the data through the network
        activatedList, summedList = self.forward_pass(inputData, self.weightList, self.biasList)

        # De Scale the data
        deScaledPredicted = self.min_max_reverser(activatedList[-1])

        # Descale the expected
        expectedData = self.min_max_reverser(expectedData)

        # Calculate the error of the network
        networkError = self.cost_function(expectedData, deScaledPredicted)

        print(f'Testing Error: {networkError.mean()}')

        # Merge date and prediction
        self.predictionDF = pd.DataFrame(np.concatenate([inputDate, deScaledPredicted], axis=2).squeeze(axis=1), columns=['DATE', 'Skelton'])


        # Split the data point into input and output, maintaining the 3D structure
        inputDataEval, inputDateEval, expectedDataEval = self.evaluationData[:, :, 2:],  self.evaluationData[:, :, 0:1], self.evaluationData[:, :, 1:2]

        # Pass the data through the network
        activatedListEval, summedListEval = self.forward_pass(inputDataEval, self.weightList, self.biasList)

        # De Scale the data
        deScaledPredictedEval = self.min_max_reverser(activatedListEval[-1])

        # Descale the expected
        expectedDataEval = self.min_max_reverser(expectedDataEval)

        # Calculate the error of the network
        networkErrorEval = self.cost_function(expectedDataEval, deScaledPredictedEval)

        print(f'Evaluation Error: {networkErrorEval.mean()}')

        # Merge date and prediction
        self.evaluationDF = pd.DataFrame(np.concatenate([inputDateEval, deScaledPredictedEval], axis=2).squeeze(axis=1), columns=['DATE', 'Skelton'])



        # Split the data point into input and output, maintaining the 3D structure
        inputDataTotal, inputDateTotal, expectedDataTotal = self.dataSet[:, :, 2:],  self.dataSet[:, :, 0:1], self.dataSet[:, :, 1:2]

        # Pass the data through the network
        activatedListTotal, summedListTotal = self.forward_pass(inputDataTotal, self.weightList, self.biasList)

        # De Scale the data
        deScaledPredictedTotal = self.min_max_reverser(activatedListTotal[-1])

        # Descale the expected
        expectedDataTotal = self.min_max_reverser(expectedDataTotal)

        # Calculate the error of the network
        networkErrorTotal = self.cost_function(expectedDataTotal, deScaledPredictedTotal)

        print(f'Total Error: {networkErrorTotal.mean()}')

        # Merge date and prediction
        self.entirePredictDF = pd.DataFrame(np.concatenate([inputDateTotal, deScaledPredictedTotal], axis=2).squeeze(axis=1), columns=['DATE', 'Skelton'])

        

    def predict_train(self):
        '''Function to predict the output of the network'''

        # Split the data point into input and output, maintaining the 3D structure
        inputData, inputDate, expectedData = self.trainingData[:, :, 2:],  self.trainingData[:, :, 0:1], self.trainingData[:, :, 1:2]

        # Pass the data through the network
        activatedList, summedList = self.forward_pass(inputData, self.weightList, self.biasList)

        # De Scale the data
        deScaledPredicted = self.min_max_reverser(activatedList[-1])

        # Descale the expected
        expectedData = self.min_max_reverser(expectedData)

        # Calculate the error of the network
        networkError = self.cost_function(expectedData, deScaledPredicted)

        # Merge date and prediction
        self.predictionDFTesting = pd.DataFrame(np.concatenate([inputDate, deScaledPredicted], axis=2).squeeze(axis=1), columns=['DATE', 'Skelton'])

        return networkError

        

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

    # Define the structure of the network
    nodeStructure = [(16, 'sigmoid'),(16, 'sigmoid'), (1, 'sigmoid')]

    # Define the learning rate and epochs
    learningRate = 0.1
    trainingEpochs = 50000

    #np.random.seed(42) 

    # Create the neural network
    neuralNetwork = MLP(dataSet, nodeStructure, learningRate, trainingEpochs, predictedColumn)

    # Train the neural network
    neuralNetwork.train_network()

    # Predict the output of the network
    neuralNetwork.predict()

    networkerror = neuralNetwork.predict_train()

 
    # Plot the data
    with PdfPages(f'plots/error_vs_epochs_{learningRate}_{trainingEpochs}.pdf') as pdf:
        line_plot_multi([neuralNetwork.epochErrorDF, neuralNetwork.epochErrorDFTesting], ['b','r'], pdf)
        scatter_plot([dataSet[dataSet['DATE'].dt.year == 1995], neuralNetwork.predictionDF], 'Skelton', pdf, ['b', 'r'])
        scatter_plot([dataSet[dataSet['DATE'].dt.year.isin([1993, 1994])], neuralNetwork.predictionDFTesting], 'Skelton', pdf, ['b', 'r'])
        scatter_plot([dataSet[dataSet['DATE'].dt.year == 1996], neuralNetwork.evaluationDF], 'Skelton', pdf, ['b', 'r'])
        scatter_plot([dataSet, neuralNetwork.entirePredictDF], 'Skelton', pdf, ['b', 'r'])
        correlation_plot([dataSet[dataSet['DATE'].dt.year == 1995], neuralNetwork.predictionDF], predictedColumn, 'Skelton', pdf, ['b', 'r'])
        correlation_plot([dataSet[dataSet['DATE'].dt.year == 1996], neuralNetwork.evaluationDF], predictedColumn, 'Skelton', pdf, ['b', 'r'])
        correlation_plot([dataSet, neuralNetwork.entirePredictDF], predictedColumn, 'Skelton', pdf, ['b', 'r'])

    print(neuralNetwork.weightList)
    print(neuralNetwork.biasList)



    for data, color in zip([neuralNetwork.epochErrorDF, neuralNetwork.epochErrorDFTesting], ['b','r']):
        plt.plot(data['Epoch'], data['Error'], c=color)
    plt.title(f'Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.grid(True)


    plt.show()


start_time = time.time()

if __name__ == '__main__':
    main()

end_time = time.time()
print(f'Time taken: {end_time - start_time} seconds')

