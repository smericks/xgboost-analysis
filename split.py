import helpers as h
import numpy as np

# uses extract_split from helpers.py to split training and testing sets, then save to .csv files


# STANDARD VERSION
"""
Xtrain, Xtest, ytrain, ytest = h.extract_split("higgs_bigger.txt", 0.2, 123)
np.savetxt('Xtrain_higgs.csv', np.array(Xtrain), delimiter=',')
np.savetxt('Xtest_higgs.csv', np.array(Xtest), delimiter=',')
np.savetxt('ytrain_higgs.csv', np.array(ytrain), delimiter=',')
np.savetxt('ytest_higgs.csv', np.array(ytest), delimiter=',')
"""

# WEIGHTED VERSION

#Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest = h.extract_split_weights("GEANT_6varcuts.txt", 0.2, 123)
Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest, Ptest = h.extract_split_polarizations("truth_inclusive.txt", 0.2, 123)
np.savetxt('Xtrain_truth.csv', np.array(Xtrain), delimiter=',')
np.savetxt('Xtest_truth.csv', np.array(Xtest), delimiter=',')
np.savetxt('ytrain_truth.csv', np.array(ytrain), delimiter=',')
np.savetxt('ytest_truth.csv', np.array(ytest), delimiter=',')
np.savetxt('Wtrain_truth.csv', np.array(Wtrain), delimiter=',')
np.savetxt('Wtest_truth.csv', np.array(Wtest), delimiter=',')
np.savetxt('Ptest_truth.csv', np.array(Ptest), delimiter=',')

