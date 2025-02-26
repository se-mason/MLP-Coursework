def delta_function(outputMatrix,currentStructure, dataOutput, learningRate):
    '''Calculate the delta function'''
    # Get the inverse activation function
    # activation = get_activation_derivative(currentStructure[1])
    inverseOutput = sigmoid_derivative(outputMatrix).flatten()

    vectorForm = outputMatrix.flatten()

    # turn the data output into a vector
    dataOutput = dataOutput * np.ones(vectorForm.shape)

    # Calculate the delta function
    delta = (dataOutput - vectorForm) * inverseOutput
    delta.reshape(outputMatrix.shape)

    return delta




def backward_pass(nodeValues, weightsAndBiases, dataOutput, learningRate, perceptronStructure):
    '''back pass algorithm'''

    weightsList, biasList = weightsAndBiases

    for i in range(0, len(nodeValues)):
        matrixPosition = (-(i+1))
        # get the matrices for the current layer
        workingMatrix = nodeValues[matrixPosition]
        workingWeights = weightsList[matrixPosition]
        workingBias = biasList[matrixPosition]
        currentStructure = perceptronStructure[1][matrixPosition]
        # calculate the delta function
        delta = delta_function(workingMatrix, currentStructure, dataOutput, learningRate).reshape(-1,1)

        '''
        # calculate the update matrices
        updateWeightMatrix = learningRate * workingMatrix * delta
        updateBiasMatrix = learningRate * delta

        # repeat the rows to math to the correct shape
        np.repeat(updateWeightMatrix, workingWeights.shape[0], axis=0)
        np.repeat(updateBiasMatrix, workingBias.shape[0], axis=0)

        # calculate the updated weights and biases
        workingWeights -= updateWeightMatrixs
        workingBias -= updateBiasMatrix'''

        print("workingMatrix shape:", workingMatrix.shape)
        print("delta shape:", delta.shape)

        updateWeightMatrix = learningRate * np.dot(workingMatrix.T, delta)
        updateBiasMatrix = learningRate * np.sum(delta, axis=0, keepdims=True)

        weightsList[matrixPosition] -= updateWeightMatrix
        biasList[matrixPosition] -= updateBiasMatrix
        
        # update the matrices
        weightsList[matrixPosition] = workingWeights
        biasList[matrixPosition] = workingBias

    return (weightsList, biasList) 