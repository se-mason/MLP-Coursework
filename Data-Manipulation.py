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
