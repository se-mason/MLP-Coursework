#Python file for reading data from a file and plotting it using matplotlib

# Cleaning the data

import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

DataBase = 'DataSet.db'
table = 'data_table'

column_names = ['Crakehill', 'Skip Bridge', 'Westwick', 'Skelton', 'Arkengarthdale', ' East Cowton', 'Malham Tarn', 'Snaizeholme' ]

# Connect to the SQLite database
conn = sql.connect(DataBase)
cursor = conn.cursor()



# Create a PdfPages object to save the plots
with PdfPages('plots.pdf') as pdf:
    # Define the SQL query to select all rows from a specific column
    for column_name in column_names:
        query = f'SELECT "index","{column_name}" FROM data_table'
        # Execute the query and fetch all results
        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert the results to a pandas DataFrame
        df = pd.DataFrame(rows, columns=['index', column_name])

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(df['index'], df[column_name], marker='o', color='b')
        plt.title(f'{column_name} Data')
        plt.xlabel('Index')
        plt.ylabel(column_name)
        plt.grid(True)        

        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()


# Close the connection
conn.close()

# Open the PDF file with the default PDF viewer
# pdf_path = 'plots.pdf'
# os.startfile(pdf_path)