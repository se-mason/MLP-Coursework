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

    # Return the column names without the first column
    return columnNames[1:]


def read_data(dataBase, dataTable, columnName):
    '''Fucntion for retreiving column data from a database, and returning a dataframe'''
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

    # Close the connection
    conn.close()

    # Return the data
    return columnData


def write_data(dataBase, dataTable, columnName, dataPoint, newValue):
    '''Function for writing data to a database'''
    # Initialise the connection and the cursor
    conn = sql.connect(dataBase)
    cursor = conn.cursor()

    # Define the SQL query to update a specific row in a specific column
    query = f'UPDATE {dataTable} SET "{columnName}" = {newValue} WHERE "DATE" = "{dataPoint[0]}"'

    # Execute the query
    cursor.execute(query)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
