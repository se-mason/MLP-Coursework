import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import reading_and_writing as rw
from matplotlib.backends.backend_pdf import PdfPages
import time

def create_data(predictedColumn: str) -> pd.DataFrame:
    '''Function to create a sample dataset'''

    # Load the data
    dataBase = rw.read_data_all('dataSet.db', 'data_table')

    # Create database for the predictors for the data
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

# Record time
start = time.time()

# Name Predictand Column
predictedColumn = 'Predicted Skelton'

# Load the data
dataSet = create_data(predictedColumn)

trainingData = dataSet[dataSet['DATE'].dt.year.isin([1993, 1994])]
testingData = dataSet[dataSet['DATE'].dt.year == 1996]

# Extract input and output data
inputData = trainingData.iloc[:, 2:].values  # Predictor columns
inputDate = trainingData.iloc[:, 0:1].values  # Date column
expectedData = trainingData.iloc[:, 1].values  # Target column (flattened)

# Same for testing data
inputData2 = testingData.iloc[:, 2:].values  # Predictor columns
inputDate2 = testingData.iloc[:, 0:1].values  # Date column
expectedData2 = testingData.iloc[:, 1].values  # Target column (flattened)

# Create a model instance and train it
model = LinearRegression()
model.fit(inputData, expectedData)

# Make predictions
new_data = testingData.iloc[:, 2:].values  # Predictor columns for testing
predictions = model.predict(new_data)

mse = np.mean((predictions - expectedData2) ** 2)
print(f"Mean Squared Error: {mse}")

# Plotting
with PdfPages(f'plots/lin_regression.pdf') as pdf:
    plt.figure(figsize=(10, 6))
    plt.scatter(inputDate2, expectedData2, color='blue', label='Data points')
    plt.plot(inputDate2, predictions, color='red', label='Regression Line')
    plt.xlabel('Date')
    plt.ylabel('Predicted Skelton')
    plt.title('Linear Regression Example')
    plt.legend()
    # Save the current figure to the PDF
    pdf.savefig()
    plt.close()

# Model coefficients
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Record time
end = time.time()
print(f"Time taken: {end - start} seconds")