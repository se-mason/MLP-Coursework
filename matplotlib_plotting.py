import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def scatter_plot(dataSets, columnName, pdfFile, colourSets):
    '''Function for plotting a scatter plot of a column of data'''
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    for data, color in zip(dataSets, colourSets):
        plt.scatter(data['DATE'], data[columnName], c=color, label=f'{color} data')
    plt.title(f'{columnName} vs time')
    plt.xlabel('Date')
    plt.ylabel(columnName)
    plt.grid(True)

    # Save the current figure to the PDF
    pdfFile.savefig()
    plt.close()

def line_plot(dataSet, pdfFile):
    '''Function for plotting a scatter plot of a column of data'''
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(dataSet['epoch'], dataSet['error'], 'b')
    plt.title(f'Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.grid(True)

    # Save the current figure to the PDF
    pdfFile.savefig()
    plt.close()

def correlation_plot(dataSets, columnX, columnY, pdfFile, colourSets):
    '''Function for plotting the correlation between two columns of data'''
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    for data, color in zip(dataSets, colourSets):
        plt.scatter(data[columnX], data[columnY], c=color, label=f'{color} data')
    plt.title(f'{columnX} vs {columnY}')
    plt.xlabel(columnX)
    plt.ylabel(columnY)
    plt.grid(True)
    plt.legend()

    # Save the current figure to the PDF
    pdfFile.savefig()
    plt.close()