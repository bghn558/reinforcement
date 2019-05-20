import numpy as np


def split(Xdata,S1data,S2data,ydata,frac=6/7.0):

    all_training_samples = int(frac * Xdata.shape[0])
    training_samples = all_training_samples
    Xtrain = Xdata[0:training_samples]
    S1train = S1data[0:training_samples]
    S2train = S2data[0:training_samples]
    ytrain = ydata[0:training_samples]

    Xtest = Xdata[all_training_samples:]
    S1test = S1data[all_training_samples:]
    S2test = S2data[all_training_samples:]
    ytest = ydata[all_training_samples:]
    # ytest = ytest.flatten()

    sortinds = np.random.permutation(training_samples)
    Xtrain = Xtrain[sortinds]
    S1train = S1train[sortinds]
    S2train = S2train[sortinds]
    ytrain = ytrain[sortinds]
    # ytrain = ytrain.flatten()
    return Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest


def process_gridworld_np_data(input):
    # run training from input npz data file, and save test data prediction in output file
    # load data from numpy file, including
    # im_data: flattened images
    # state_data: concatenated one-hot vectors for each state variable
    # state_xy_data: state variable (x,y position)
    # label_data: one-hot vector for action (state difference)
    arr_data = np.load(input)
    Xdata = arr_data["arr_0"]

    state1_data = arr_data["arr_1"]
    state2_data = arr_data["arr_2"]
    label_data = arr_data["arr_3"]
    ydata = label_data.astype('int8')
    Xdata = Xdata.astype('float32')

    S1data = state1_data.astype('int8')
    S2data = state2_data.astype('int8')
    return Xdata,S1data,S2data,ydata
