#based off of this tutorial: https://www.datacamp.com/community/tutorials/xgboost-in-python

import numpy as np
import helpers as h
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)

#global user-defined variables

# use split.py first to create necessary files
Xtrain = np.loadtxt('Xtrain_geant.csv', delimiter=',')
Xtest = np.loadtxt('Xtest_geant.csv', delimiter=',')
ytrain = np.loadtxt('ytrain_geant.csv', delimiter=',')
ytest = np.loadtxt('ytest_geant.csv', delimiter=',')
Wtrain = np.loadtxt('Wtrain_geant.csv', delimiter=',')
Wtest = np.loadtxt('Wtest_geant.csv', delimiter=',')	
Ptest = np.loadtxt('Ptest_geant.csv', delimiter=',')

if __name__=="__main__":
	
	#reweight training for same contribution from signal and background
	Wtrain = h.reweight_all(ytrain, Wtrain)

	#define model
	#xg_mod = xgb.XGBClassifier(objective = objective, max_depth = max_depth, alpha = alpha, n_estimators = n_estimators)
	xg_mod = xgb.XGBClassifier(objective = 'binary:logistic')
	
	#training
	#xg_mod.fit(Xtrain,ytrain, sample_weight=Wtrain)
		
	#early stopping rounds training
	eval_set = [ (Xtest, ytest) ]
	xg_mod.fit(Xtrain, ytrain, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True, sample_weight=Wtrain)

	#testing
	preds = xg_mod.predict(Xtest)
	
	print("Percent Accuracy: ", h.eval_preds(ytest,preds) )

	#plot trees	
	#h.plot_trees(xg_mod, 11)

	#plot feature importance
	h.plot_importance(xg_mod)
		
	#plot ROC and calculate AUC
	h.plot_ROC(xg_mod, Xtest, ytest)

	# probability distribution plot
	#h.best_threshold(xg_mod, Xtest, ytest)
	#h.output_threshold_plot(xg_mod, Xtest, ytest)
	h.output_threshold_plot(xg_mod, Xtest, ytest, True, Wtest)
	h.output_threshold_plot(xg_mod, Xtest, ytest, True, Wtest, True, Ptest)
	Wtest = h.reweight_unity(Wtest, Ptest)
	h.output_threshold_plot(xg_mod, Xtest, ytest, True,  Wtest, True, Ptest)
