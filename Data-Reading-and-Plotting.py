#Python file for reading data from a file and plotting it using matplotlib

# Cleaning the data

import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt

DataBase = 'DataSet.db'
table = 'data_table'

column_names = ['Crakehill', 'Skip Bridge', 'Westwick', 'Skelton', 'Arkengarthdale', ' East Cowton', 'Malham Tarn', 'Snaizeholme' ]

# Connect to the SQLite database
conn = sql.connect(DataBase)
cursor = conn.cursor()

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
    plt.show()
    plt.pause(0.001)  # Pause to ensure the plot is updated
    plt.close()

    




# Convert the results to a pandas DataFrame
df = pd.DataFrame(rows, columns=[column_name])

# Display the DataFrame
print(df)

# Close the connection
conn.close()
