import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def scatter_plot(columnName, columnData, pdfPath, pointColour):
    '''Function for plotting a scatter plot of a column of data'''
    # Create a PdfPages object to save the plots
    with PdfPages(pdfPath) as pdf:
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(columnData.index, columnData[columnName], marker='o', linestyle='-', color=pointColour)
        plt.title(f'{columnName} vs time')
        plt.xlabel('date')
        plt.ylabel(columnName)
        plt.grid(True)
        
        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()

