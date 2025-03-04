import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import reading_and_writing as rw

dataBase = 'DataSet.db'
dataTable = 'data_table'

# Read the data
dataFrame = rw.read_data_all(dataBase, dataTable).iloc[:, 1:]

# Shift the data 
dataFrame[f'Skelton + 1'] = dataFrame['Skelton'].shift(-1)

# Add a shifted rainfall column
rainfallColumns = ['Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme']
for column in rainfallColumns:
    for i in range(1, 3):
        dataFrame[f'{column} - {i}'] = dataFrame[column].shift(i)

# Drop na values
dataFrame = dataFrame.dropna()

# Calculate the correlation matrix
correlationMatrix = dataFrame.corr()

# Sort the correlation matrix
correlationMatrix = correlationMatrix.sort_values(by='Skelton + 1', axis=0, ascending=True)
correlationMatrix = correlationMatrix.sort_values(by='Skelton + 1', axis=1, ascending=True)


# Plot the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()