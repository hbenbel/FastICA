import argparse
import os

import numpy as np
from scipy.io import wavfile


# Function to load the audio data
def loadData(dataPath, nsources=9, size=50000):
    data = np.empty((nsources, size), np.float32)

    for i in range(nsources):
        fs, data[i,:] = wavfile.read(os.path.join(dataPath, 'mix') + str(i + 1)  + '.wav')

    return fs, data

# Function to save the audio data separated by fastica
def saveData(saveDataPath, fs, X):
    for i in range(X.shape[0]):
        wavfile.write(os.path.join(saveDataPath, 'ica_') + str(i + 1) + ".wav", fs, X[i,:])

# Definition of G'
def Gprime(x):
    return np.tanh(x)

# Definition of G''
def Gsecond(x):
    return np.ones(x.shape) - np.power(np.tanh(x), 2)

# Center matrix X
def centerMatrix(X, N):
    mean = X.mean(axis=1)
    M = X - (mean.reshape((N, 1)) @ np.ones((1, X.shape[1])))
    return M

# Whiten matrix X with eigenvalue decomposition
def whitenMatrix(X):
    D, E = np.linalg.eigh(X @ X.T)
    DE = E @ np.diag(1/np.sqrt(D + 1e-5)) @ E.T
    
    return DE @ X

# One-unit algorithm step
def oneUnit(X, wp):
    term1 = np.mean((X @ Gprime(wp @ X).T), axis=1)
    term2 = np.mean(Gsecond(wp @ X), axis=1) * wp
    return term1 - term2

# Deflationary orthogonalization
def orthogonalize(W, wp, i):
    return wp - ((wp @ W[:i,:].T) @ W[:i,:])

# wp normalization
def normalize(wp):
    return wp / np.linalg.norm(wp)

# Function to see if wp is still updating
def diff(wp1, wp2):
    norm1 = np.linalg.norm(wp1)
    norm2 = np.linalg.norm(wp2)
    return np.abs(norm1 - norm2)


def fastICA(X, C, max_iter):
    N = X.shape[0]
    X = centerMatrix(X, N)
    X = whitenMatrix(X)
    
    W = np.zeros((C, N))
    for i in range(C):
        wp = np.random.rand(1, N)
        for j in range(max_iter):
            old_wp = wp
            wp = oneUnit(X, wp)
            wp = orthogonalize(W, wp, i)
            wp = normalize(wp)
            W[i,:] = wp

    return W @ X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastICA to solve the cocktail party problem')
    parser.add_argument('--dataPath', '-d', type=str, help='Path to the folder containing the input audio files', required=True)
    parser.add_argument('--saveDataPath', '-s', type=str, help='Path to folder that will contain the output audio files', required=True)
    parser.add_argument('--nComponents', '-n', type=int, help='Number of components that we want', required=True)
    parser.add_argument('--max_iter', '-m', type=int, default=200, required=False, help='Number of iteration to get a component')

    args = parser.parse_args()
    dataPath = args.dataPath
    saveDataPath = args.saveDataPath
    nComponents = args.nComponents
    max_iter = args.max_iter

    if not os.path.exists(saveDataPath):
        os.makedirs(saveDataPath)

    fs, X = loadData(dataPath)
    S = fastICA(X, nComponents, max_iter)
    saveData(saveDataPath, fs, S)
