import pandas as pd
import numpy as np
import sqlite3 
from sqlalchemy import create_engine


def data_import(dataBase: pd.DataFrame, dataTable: str, excelFile: str):
    '''Function for importing Excel data into a database'''

    # Read the excel file and store the data in a DataFrame
    excelFile = pd.read_excel(excelFile)

    # Check for non-numeric values in non-first columns and replace them with na points
    excelFile.iloc[:, 1:] = excelFile.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(np.nan)

    # Create a connection to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(dataBase)
    cursor = conn.cursor()

    # Drop the table if it exists so the database can be recreated from scratch for testing
    cursor.execute(f'DROP TABLE IF EXISTS {dataTable}')

    # Use SQLAlchemy to import the DataFrame into the SQL database
    engine = create_engine(f'sqlite:///{dataBase}')
    excelFile.to_sql(dataTable, engine, if_exists='replace', index=False)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    print('Data imported successfully!')

