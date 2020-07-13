#based off of this tutorial: https://www.datacamp.com/community/tutorials/xgboost-in-python

import numpy as np
import helpers as h
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#global user-defined variables

# use split.py first to create necessary files
Xtrain = np.loadtxt('Xtrain_higgs.csv', delimiter=',')
Xtest = np.loadtxt('Xtest_higgs.csv', delimiter=',')
ytrain = np.loadtxt('ytrain_higgs.csv', delimiter=',')
ytest = np.loadtxt('ytest_higgs.csv', delimiter=',')

if __name__=="__main__":

	#define model
	xg_mod = xgb.XGBClassifier(objective = 'binary:logistic')
	
	#train & test
	xg_mod.fit(Xtrain,ytrain)
	preds = xg_mod.predict(Xtest)

	print("Percent Accuracy: ", h.eval_preds(ytest,preds) )

	#plot trees	
	#h.plot_trees(xg_mod, 11)

	#plot feature importance
	h.plot_importance(xg_mod)
		
	#plot ROC and calculate AUC
	np.set_printoptions(threshold=np.inf)
	h.plot_ROC(xg_mod, Xtest, ytest)

	#probability distribution  plot
	h.output_threshold_plot(xg_mod, Xtest, ytest)


