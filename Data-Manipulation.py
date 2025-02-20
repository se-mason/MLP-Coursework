# Function for putting all datapoints within a standard range

import pandas as pd
import sqlite3 as sql

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