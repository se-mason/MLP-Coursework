# Function for putting all datapoints within a standard range

import pandas as pd
import sqlite3 as sql
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from data_reading_and_plotting import plotGraph
import os


DataBase = 'DataSet.db'
table = 'data_table'
# Connect to the SQLite database
conn = sql.connect(DataBase)
cursor = conn.cursor()

# Define the SQL query to get all column names from the table
query = f"PRAGMA table_info({table})"

# Execute the query and fetch all results
cursor.execute(query)
columns_info = cursor.fetchall()

# Extract column names from the results
column_names = [info[1] for info in columns_info]

# Remove the first two columns (index and date)
column_names = column_names[2:]


def boundsFunction(DF_range, deviationWeight):
    '''Fucntion to calcualte the upper and lowerbounds of accepted values'''

    # Calculate the mean and standard deviation of the DataFrame
    deviation = DF_range.std(axis=0, skipna=True, ddof=1, numeric_only=True)
    average = DF_range.mean(axis=0, skipna=True, numeric_only=True)

    # Calculate the upper and lower bounds
    lower_bound = average.iloc[1] - (deviationWeight * deviation.iloc[1])
    upper_bound = average.iloc[1] + (deviationWeight * deviation.iloc[1])

    # Return the bounds
    return lower_bound, upper_bound

def pointReplacer(neighborRange, DFLength, rangeFrame, point, removedDict):
    '''Function to replace a point with suitable data'''
    sideRange = round(neighborRange/2)

    # Checks if the data is out of bounds
    # if the point is close to 0
    if int(point.iloc[0])-sideRange < 0:
            start = 0
    else:
            start = int(point.iloc[0])-sideRange

    # if the point is close to the end of the DF
    if int(point.iloc[0])+sideRange > DFLength:
        end = DFLength
    else:
        end = int(point.iloc[0])+sideRange


    


def removeOutliers(columnDF, column_name, remove_count, removedDict):
    '''Function to remove outliers from a DataFrame'''

    # Define dictionary of highest flow rate/rainfall recorded ever for each location
    max_values = {
        'Crakehill': 239, #27/09/2012 https://environment.data.gov.uk/hydrology/station/d1a229c2-1eec-43dd-8b89-1999568e293c
        'Skip Bridge': 247, #26/12/2015 https://environment.data.gov.uk/hydrology/station/aaac507d-5e62-474f-815d-c0b46aea04e8
        'Westwick': 518, #6/12/2015 https://environment.data.gov.uk/hydrology/station/e2db454d-9cdd-43b0-b47c-6fb047bd6fb4
        'Skelton': 583, #4/01/1982 https://environment.data.gov.uk/hydrology/station/213d70b2-894b-406b-9dc3-31d3ccec7f54
        'Arkengarthdale': 107.8, #30/07/2019 https://environment.data.gov.uk/hydrology/station/d5a68302-2d66-400f-bff8-43d09ea31204
        'East Cowton': 79.6, #24/09/2012 https://environment.data.gov.uk/hydrology/station/cf79b9c5-553c-4bd5-a8de-0974956a521b
        'Malham Tarn': 85.2, #30/07/2019 https://environment.data.gov.uk/hydrology/station/8ec41d7f-cb69-4d72-beb1-1e8cc4c5a50f
        'Snaizeholme': 129.6, #07/01/2005,19/02/1990  (Tow Hill)https://environment.data.gov.uk/hydrology/station/9f9631fd-c5f1-4cef-a779-b3a3782b26b2
    }


    # Adds the column to the removed list
    removedDict[column_name] = []

    # As these max flow rates are the past, we should remove any values above these * by a constant factor
    outlierContsant = 3

    # Determines if each point is an outlier
    for point in columnDF.itertuples():
        print(point)
        print(point[3])
        # No point should be negative
        # No point should be negative
        if int(point[3]) < 0:
            columnDF.loc[point.Index, column_name] = np.nan
            remove_count += 1
            removedDict[column_name].append([point.Index, point.DATE, point[3]]) # Stores the data point that is removed
            continue

        # Max value checks
        if int(point[3]) > (max_values[column_name] * outlierContsant):
            columnDF.loc[point.Index, column_name] = np.nan
            remove_count += 1
            removedDict[column_name].append([point.Index, point.DATE, point[3]]) # Stores the data point that is removed
            continue

    
    return remove_count, removedDict, columnDF


# remove points to na
# then replace na points but use skip na to get the average of the points around it

def standardiseData():
    '''Function to standardise all data points in the database'''
    # counter to keep track of the number of points removed
    remove_count = 0
    # dictionary to store the removed data points
    removedDict = {}

    # Create a PdfPages object to save the plots
    with PdfPages('proposed_plots.pdf') as pdf:
        for column in column_names:
            # retrieves the data from the column
            query = f'SELECT "index","DATE", "{column}" FROM data_table'
            cursor.execute(query)
            rows = cursor.fetchall()
            columnDF = pd.DataFrame(rows, columns=['index', 'DATE', column])

            # remove the outliers from the set and replace with NaN
            remove_count, removedDict, columnDF = removeOutliers(columnDF, column, remove_count, removedDict)

            # plot the graph
            plotGraph(columnDF, column, pdf)

    pdf_path = 'proposed_plots.pdf'
    os.startfile(pdf_path)
    print(removedDict)




standardiseData()
# Close the connection 
conn.close()
