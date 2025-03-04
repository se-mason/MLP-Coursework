import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import reading_and_writing as rw

dataBase = 'DataSet.db'
dataTable = 'data_table'

# Read the data
dataFrame = rw.read_data_all(dataBase, dataTable).iloc[:, 1:]

# Create a new table to show correlation
correlationTable = pd.DataFrame()

# Shift the data 
dataFrame[f'Skelton + 1'] = dataFrame['Skelton'].shift(-1)


# Add a shifted rainfall column
rainfallColumns = ['Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme']
for column in rainfallColumns:
    for i in range(1, 3):
        dataFrame[f'{column} - {i}'] = dataFrame[column].shift(i)

flowColumns = ['Skelton', 'Westwick', 'Skip Bridge', 'Crakehill']
for column in flowColumns:
    dataFrame[f'{column} - 1'] = dataFrame[column].shift(1)

# Testing Correlations
dataFrame['Rainfall average'] = (dataFrame['Arkengarthdale'] + dataFrame['East Cowton'] + dataFrame['Malham Tarn'] + dataFrame['Snaizeholme']) / 4
dataFrame['Rainfall average -1'] = (dataFrame['Arkengarthdale - 1'] + dataFrame['East Cowton - 1'] + dataFrame['Malham Tarn - 1'] + dataFrame['Snaizeholme - 1']) / 4
dataFrame['Rainfall average -2'] = (dataFrame['Arkengarthdale - 2'] + dataFrame['East Cowton - 2'] + dataFrame['Malham Tarn - 2'] + dataFrame['Snaizeholme - 2']) / 4
dataFrame['Rainfall average average'] = (dataFrame['Rainfall average'] + dataFrame['Rainfall average -1'] + dataFrame['Rainfall average -2']) 
dataFrame['Flow average'] = (dataFrame['Skelton'] + dataFrame['Westwick'] + dataFrame['Skip Bridge'] + dataFrame['Crakehill']) / 4
dataFrame['Flow average -1'] = (dataFrame['Skelton - 1'] + dataFrame['Westwick - 1'] + dataFrame['Skip Bridge - 1'] + dataFrame['Crakehill - 1']) / 4
dataFrame['Skelton + rain'] = dataFrame['Skelton'] + (dataFrame['Rainfall average'] + dataFrame['Rainfall average -1']*0.7 + dataFrame['Rainfall average -2']*0.3)/2
dataFrame['Skelton + rain 2'] = (dataFrame['Skelton'] + dataFrame['Rainfall average'])   
dataFrame['Skelton + rain 3'] = (dataFrame['Skelton - 1'] + dataFrame['Rainfall average -1'])
dataFrame['Skelton + rain 4'] = (dataFrame['Skelton'] + dataFrame['Rainfall average average'])
dataFrame['Skelton + rain 5'] = ((dataFrame['Skelton']+ dataFrame['Skelton - 1'])/2 + dataFrame['Rainfall average average'])
dataFrame['average + rain'] = (dataFrame['Flow average'] + dataFrame['Rainfall average'])   
dataFrame['average + rain 2'] = (dataFrame['Flow average'] + dataFrame['Rainfall average -1'])
dataFrame['average + rain 3'] = (dataFrame['Flow average'] + dataFrame['Rainfall average average'])
dataFrame['average + rain 4'] = (dataFrame['Flow average -1'] + dataFrame['Rainfall average -1'])
dataFrame['Normalized Rainfall'] = (dataFrame['Rainfall average'] - dataFrame['Rainfall average'].mean()) / dataFrame['Rainfall average'].std()
dataFrame['Normalized Flow'] = (dataFrame['Flow average'] - dataFrame['Flow average'].mean()) / dataFrame['Flow average'].std()

# Plot chosen corelations
correlationTable['Skelton + 1'] = dataFrame['Skelton'].shift(-1)
correlationTable['Average flow  + Average Rain'] = (dataFrame['Flow average'] + dataFrame['Rainfall average'])
correlationTable['Skelton + average rain'] = (dataFrame['Skelton'] + dataFrame['Rainfall average'])
correlationTable['Flow average'] = (dataFrame['Skelton'] + dataFrame['Westwick'] + dataFrame['Skip Bridge'] + dataFrame['Crakehill']) / 4
correlationTable['Westwick'] = (dataFrame['Westwick'])
correlationTable['Skelton'] = dataFrame['Skelton']
correlationTable['Rainfall averaged over 3 days'] = (dataFrame['Rainfall average'] + dataFrame['Rainfall average -1'] + dataFrame['Rainfall average -2'])




# Drop na values
correlationTable = correlationTable.dropna()

# Calculate the correlation matrix
correlationMatrix = correlationTable.corr()

# Sort the correlation matrix
correlationMatrix = correlationMatrix.sort_values(by='Skelton + 1', axis=0, ascending=True)
correlationMatrix = correlationMatrix.sort_values(by='Skelton + 1', axis=1, ascending=True)


# Plot the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()