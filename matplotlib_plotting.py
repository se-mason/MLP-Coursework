import pandas as pd
import numpy as np
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
    plt.plot(dataSet['Epoch'], dataSet['Error'], 'b')
    plt.title(f'Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.grid(True)

    # Save the current figure to the PDF
    pdfFile.savefig()
    plt.close()


def line_plot_multi(dataSets, colourSets, pdfFile):
    '''Function for plotting a scatter plot of a column of data'''
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    for data, color in zip(dataSets, colourSets):
        plt.plot(data['Epoch'], data['Error'], c=color)
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
    line = np.linspace(1, 200, 200)
    plt.scatter(dataSets[0][columnX], dataSets[1][columnY], c=colourSets[0], marker='.')
    plt.plot(line,line, color=colourSets[1], label='Ideal Values')
    plt.xlabel(f'{columnX} Actual')
    plt.ylabel(f'{columnY} Predicted')
    plt.legend()
    plt.title(f'Correlation Plot')

    # Save the current figure to the PDF
    pdfFile.savefig()
    plt.close()

def plot_text(text, pdfFile):
    '''Function for plotting text'''
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, text, fontsize=12, ha='center')
    plt.axis('off')

    # Save the current figure to the PDF
    pdfFile.savefig()
    plt.close()
 