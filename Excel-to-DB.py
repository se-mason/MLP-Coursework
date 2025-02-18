# File for importing Excel data into a database
import pandas as pd
import sqlite3 
from sqlalchemy import create_engine
 

# Read the excel file and store the data in a DataFrame
data = pd.read_excel('Data-NoHeads.xlsx')

# Create a connection to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('DataSet.db')
cursor = conn.cursor()

# Create a table (adjust the schema as needed)
cursor.execute('''
CREATE TABLE IF NOT EXISTS data_table (
    column1 TEXT,
    column2 TEXT,
    column3 TEXT
)
''')

# Use SQLAlchemy to import the DataFrame into the SQL database
engine = create_engine('sqlite:///DataSet.db')
data.to_sql('data_table', engine, if_exists='replace', index=False)

# Commit the changes and close the connection
conn.commit()
conn.close()