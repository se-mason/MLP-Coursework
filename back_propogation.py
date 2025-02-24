# back Propogation algorithm
import numpy as np

perceptronStructure = (4,3,1)
learningRate = 0.1
weightList = [0.1, 0.6, 0.5, 0.9, 0.1]
biasValue = 0

testData = [

    [0.5, 1.2, 3.4, 2.1, 0.7],
    [2.3, 3.1, 0.8, 1.5, 1.1],
    [1.1, 0.9, 2.5, 3.3, 0.9],
    [3.5, 2.7, 1.4, 0.6, 1.3],
    [0.7, 1.9, 3.8, 2.5, 0.8],
    [2.8, 3.4, 1.1, 0.9, 1.2],
    [1.5, 0.6, 2.9, 3.7, 1.0],
    [3.2, 2.2, 1.3, 0.8, 1.4],
    [0.9, 1.5, 3.6, 2.0, 0.7],
    [2.6, 3.2, 1.0, 1.2, 1.1],
    [1.0, 0.8, 2.7, 3.5, 0.9],
    [3.7, 2.9, 1.2, 0.7, 1.3],
    [0.6, 1.8, 3.9, 2.4, 0.8],
    [2.9, 3.5, 1.3, 1.0, 1.2],
    [1.4, 0.7, 2.8, 3.6, 1.0],
    [3.1, 2.1, 1.5, 0.9, 1.4],
    [0.8, 1.6, 3.7, 2.3, 0.7],
    [2.7, 3.3, 1.2, 1.1, 1.1],
    [1.2, 0.9, 2.6, 3.4, 0.9],
    [3.6, 2.8, 1.1, 0.7, 1.3],
]




def activation_function(x):
    return 1 / (1 + np.exp(-x))

def initMLP(perceptronStructure):
    None

def forward_pass(dataRow, weightList, biasValue):
    None    

def backward_pass():
    Nonesz