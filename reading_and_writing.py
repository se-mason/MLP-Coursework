import sqlite3 as sql
import pandas as pd


def column_names(dataBase, dataTable):
    '''Function for retireving column names from a table in a database'''
    # Initialise the connection and the cursor
    conn = sql.connect(dataBase)
    cursor = conn.cursor()

    # Define the SQL query to get all column names from the table
    query = f"PRAGMA table_info({dataTable})"
    # Execute the query and fetch all results
    cursor.execute(query)
    columnInfo = cursor.fetchall()

    # Extract column names from the results
    columnNames = [info[1] for info in columnInfo]

    # Close the connection
    conn.close()
    print(columnNames)

    # Return the column names without the first column
    return columnNames[1:]


def read_data(dataBase, dataTable, columnName):
    '''Function for retrieving column data from a database, and returning a dataframe'''
    # Connect to the SQLite database
    conn = sql.connect(dataBase)
    cursor = conn.cursor()

    # Define the SQL query to select all rows from a specific column
    query = f'SELECT "DATE", "{columnName}" FROM {dataTable}'

    # Execute the query and fetch all results
    cursor.execute(query)
    rowData = cursor.fetchall()

    # Convert the results to a pandas DataFrame
    columnData = pd.DataFrame(rowData, columns=['DATE', columnName])

    # Convert the 'DATE' column to datetime format
    columnData['DATE'] = pd.to_datetime(columnData['DATE'])

    # Close the connection
    conn.close()

    # Return the data
    return columnData

def read_data_all(dataBase, dataTable):
    '''Function for retrieving all column data from a database, and returning a dataframe'''
    # Connect to the SQLite database
    conn = sql.connect(dataBase)
    cursor = conn.cursor()

    # Define the SQL query to select all rows from a specific column
    query = f'SELECT * FROM {dataTable}'

    # Execute the query and fetch all results
    cursor.execute(query)
    rowData = cursor.fetchall()

    # Convert the results to a pandas DataFrame
    dataBase = pd.DataFrame(rowData, columns=['DATE', 'Crakehill', 'Skip Bridge', 'Westwick', 'Skelton', 'Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme'])

    # Convert the 'DATE' column to datetime format
    dataBase['DATE'] = pd.to_datetime(dataBase['DATE'])

    # Close the connection
    conn.close()

    # Return the data
    return dataBase

def write_data(dataBase, dataTable, columnName, dataPoints):
    '''Function for writing data to a database'''
    # Initialise the connection and the cursor
    conn = sql.connect(dataBase)
    cursor = conn.cursor()

    ## Prepare the SQL query for batch updates
    query = f'UPDATE {dataTable} SET "{columnName}" = ? WHERE rowid = ?'

    # Execute the query for each data point
    cursor.executemany(query, [(dataPoint[2], int(dataPoint[0])+1) for dataPoint in dataPoints])

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def read_dictionary(nestedDict, columnName, posIndex):
    '''function to create a dataframe from a nested dictionary'''

    # Filter dictionary
    filtered_dict = {k: v[posIndex] for k, v in nestedDict.items()}

    # Convert the filtered dictionary to a DataFrame
    newData = pd.DataFrame(list(filtered_dict.items()), columns=['DATE', columnName])

    # Convert the 'DATE' column to datetime format
    newData['DATE'] = pd.to_datetime(newData['DATE'])

    return newData
