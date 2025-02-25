import numpy as np
import pandas as pd

def local_range_df(columnData:pd.DataFrame, dataPoint:tuple, columnLength:int, windowSize:int) -> pd.DataFrame:
    '''Function to create a smaller range of data around a point'''
    windowSplit = round(windowSize/2)

    # Defines the start and end of the range
    if int(dataPoint[0])-windowSplit < 0:
        '''If the point is close to 0'''
        start = 0
        end = windowSize-int(dataPoint[0])
    elif int(dataPoint[0])+windowSplit > columnLength:
        '''If the point is close to the end of the DF'''
        end = columnLength
        start = columnLength - windowSize
    else:
        '''If the point is in the middle of the DF'''
        start = int(dataPoint[0])-windowSplit
        end = int(dataPoint[0])+windowSplit

    # Create a range of the data around the point
    localData = columnData.loc[start:end]

    return localData


def standard_deviation_calculator(columnData:pd.DataFrame, columnName:str, deviationWeight:int) -> tuple[float, float]:
    '''Function to calculate the standard deviation of a range of data'''

    # Calculate the mean and standard deviation of the DataFrame
    deviation = columnData[columnName].std(axis=0, skipna=True, numeric_only=True, ddof=1)
    average = columnData[columnName].mean(axis=0, skipna=True, numeric_only=True)

    # Calculate the upper and lower bounds
    lowerBound = average - (deviationWeight * deviation)
    upperBound = average + (deviationWeight * deviation)

    return lowerBound, upperBound


def iqr_calculator(columnData: pd.DataFrame, columnName: str, deviationWeight: int) -> tuple[float, float]:
    '''Function to calculate the IQR of a range of data'''

    # Calculate the IQR of the range
    Q1 = columnData[columnName].quantile(0.25)
    Q3 = columnData[columnName].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the upper and lower bounds, round to 3 decimal places
    lowerBound = round(Q1 - (deviationWeight * IQR), 3)
    upperBound = round(Q3 + (deviationWeight * IQR), 3)

    return lowerBound, upperBound


# Complex Functions

def bounds_check_and_replace(columnData, columnName, dataPoint, iqrWeight, sdWeight, windowSize, removedCount, removedDict, updatedDict):
    '''Function to check if a data point is an outlier'''

    # Define which bounds to use for each column
    dataType = {
        'Crakehill': 'Flow-Rate', 
        'Skip Bridge': 'Flow-Rate', 
        'Westwick': 'Flow-Rate', 
        'Skelton': 'Flow-Rate',  
        'Arkengarthdale': 'Rain-Fall',
        'East Cowton': 'Rain-Fall', 
        'Malham Tarn': 'Rain-Fall', 
        'Snaizeholme': 'Rain-Fall',}
    
    # Get the length of the column
    columnLength = len(columnData)
    
    # Get the local range of data around the point
    localData = local_range_df(columnData, dataPoint, columnLength, windowSize)
    
    # Get the bounds for the data point
    if dataType[columnName] == 'Rain-Fall':
        '''If the data is rain fall data, use the IQR bounds'''
        lowerBound, upperBound = iqr_calculator(localData, columnName, iqrWeight)
    else:
        '''If the data is flow rate data, use the standard deviation bounds'''
        lowerBound, upperBound = standard_deviation_calculator(localData, columnName, sdWeight)

    # Check if the data point is an outlier
    if not (lowerBound <= dataPoint[2] <= upperBound):
        '''Removes the data point if it is an outlier'''
        # Sets point to NaN
        columnData.loc[dataPoint[0], columnName] = np.nan
        # Updates the statistics 
        removedCount += 1
        removedDict[columnName][dataPoint[1]]= [dataPoint[0], dataPoint[2]]
        updatedDict[columnName][dataPoint[1]]= [dataPoint[0], dataPoint[2]]

    return columnData, removedCount, removedDict, updatedDict
       

def extreme_value_remover(columnData, columnName, deviationWeight, removedCount, removedDict):
    '''Function to remove extreme values from a DataFrame'''

    for dataPoint in columnData.itertuples():
        '''Iterate through the data points in the DataFrame'''

        if dataPoint[2] < 0:
            '''If the data point is less than 0, replace it with a NaN'''
            columnData.loc[dataPoint[0], columnName] = np.nan
            # Increment Counter and track data
            removedCount += 1
            removedDict[columnName][dataPoint[1]]= [dataPoint[0], dataPoint[2]]

        # Calculate IQR upper bound for extreme ouliers 
        lower_bound, upper_bound = iqr_calculator(columnData, columnName, deviationWeight)

        if dataPoint[2] > upper_bound:
            '''If the data point is greater than the bound, replace it with a NaN'''
            columnData.loc[dataPoint[0], columnName] = np.nan
            # Increment Counter and track data
            removedCount += 1
            removedDict[columnName][dataPoint[1]]= [dataPoint[0], dataPoint[2]]
            

    # return the modified DataFrame
    return columnData, removedCount, removedDict

 
def simple_moving_average(columnData:pd.DataFrame, columnName:str, windowSize:int, removedDict:dict, updatedDict:dict) -> tuple[dict, dict]:
    '''Function to update empty points in the dataset'''

    # Find all NaN values in the column
    emptyPoints = columnData[columnData[columnName].isna()]

    # Iterate through the empty points
    for point in emptyPoints.itertuples():
        '''Iterate through the empty points in the DataFrame'''

        # Get the local range of data around the point
        localData = local_range_df(columnData, point, len(columnData), windowSize)

        # Calculate the average of the range
        average = round(localData[columnName].mean(skipna=True),)

        # Update the statistics
        removedDict[columnName][point[1]].append(float(average))

        # For visualisation purposes, store some of the updated data in separate dictionary
        try:
            updatedDict[columnName][point[1]].append(float(average))
        except:
            None

        # Replace the empty point with the average
        columnData.loc[point[0], columnName] = average

    return removedDict, updatedDict