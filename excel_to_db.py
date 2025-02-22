# File for importing Excel data into a database
import pandas as pd
import sqlite3 
from sqlalchemy import create_engine
import openpyxl
 

# Read the excel file and store the data in a DataFrame
data = pd.read_excel('Data-NoHeads.xlsx')

# Check for non-numeric values in non-first columns and replace them with -100
data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(-100)

# Modify the index to be 1-based
# data.index += 1

# Create a connection to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('DataSet.db')
cursor = conn.cursor()

# Drop the table if it exists so I can easily recreate the DB from scratch
cursor.execute('DROP TABLE IF EXISTS data_table')

# Dynamically create the table schema based on the DataFrame's columns

create_table_query = f'''
CREATE TABLE data_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT
)
'''

# Use SQLAlchemy to import the DataFrame into the SQL database
engine = create_engine('sqlite:///DataSet.db')
data.to_sql('data_table', engine, if_exists='replace', index=True, index_label='index')

# Commit the changes and close the connection
conn.commit()
conn.close()
print('Data imported successfully!')