# Main run file for the program
from excel_import import data_import
from reading_and_writing import column_names, read_data, write_data, read_dictionary
from data_cleaning import iqr_calculator, extreme_value_remover
from matplotlib_plotting import scatter_plot
from matplotlib.backends.backend_pdf import PdfPages
import data_cleaning as dc

removedCount = 0
removedDict = {}
updatedDict = {}

dataBase = 'DataSet.db'
dataTable = 'data_table'
excelFile = 'xlsx/Data-NoHeads.xlsx'
EXdeviationWeight = 10

iqrWeight = 1.5
sdWeight = 2
windowSize = 10
replacementRange = 6


pdfPath = f'plots/iqr{iqrWeight}_sd{sdWeight},ws{windowSize}_rr{replacementRange}.pdf'
#pdfPath = 'plots/outliers_excluded.pdf'

# import into database
data_import(dataBase, dataTable, excelFile)

# retreive column names
columnNames = column_names(dataBase, dataTable)


# open pdf to save plots
with PdfPages(pdfPath) as pdfFile:
    for columnName in columnNames:
        print(f'Current column: {columnName}')

        # collect data
        columnData = read_data(dataBase, dataTable, columnName)

        # add to removed dict
        removedDict[columnName] = {}
        updatedDict[columnName] = {}

        # remove extreme values
        columnData, removedCount, removedDict = extreme_value_remover(columnData, columnName, EXdeviationWeight, removedCount, removedDict)
        #scatter_plot([columnData], columnName, pdfFile, ['b'])

        # iterate through data
        for dataPoint in columnData.itertuples():

            # Remove extreme values
            columnData, removedCount, removedDict, updatedDict = dc.bounds_check_and_replace(columnData, columnName, dataPoint, iqrWeight, sdWeight, windowSize, removedCount, removedDict, updatedDict)

        # Replace outliers with accurate data
        removedDict, updatedDict = dc.simple_moving_average(columnData, columnName, replacementRange, removedDict, updatedDict)

        # Plot data
        # Removed data
        removedData = read_dictionary(updatedDict[columnName], columnName, 1)
        # Replaced data
        replacedData = read_dictionary(updatedDict[columnName], columnName, 2)


        # Plot all data on the same plot
        scatter_plot([columnData, removedData, replacedData], columnName, pdfFile, ['b', 'r', 'g'])

print(f"Removed {removedCount} values")

if input('Would you like to save this outcome [y/n]: ') == 'y':
    # Write the data back to the database
    for columnName in removedDict:
        dataPoints = [removedDict[columnName][dataPoint] for dataPoint in removedDict[columnName]]
        print(f"Updating {len(dataPoints)} data points for column: {columnName}")
        write_data(dataBase, dataTable, columnName, dataPoints)


print('Data saved to database')   


# post ploting
with PdfPages('plots/plots_check.pdf') as pdf:
    for columnName in columnNames:
        print(f'Current column: {columnName}')

        # collect data
        columnData = read_data(dataBase, dataTable, columnName)

        # Plot data
        scatter_plot([columnData], columnName, pdf, ['b'])