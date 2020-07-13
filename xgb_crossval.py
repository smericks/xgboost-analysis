# custom version of 3-fold cross validation for xgboost
# allows for training events to be reweighted, and for results to be concatenated together for output plots
import numpy as np
import helpers as h
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# set globals
np.set_printoptions(threshold=np.inf)
file_name = "GEANT_polar_allcuts.txt"


# helper function for train_model()
def split_columns(dataset) :
	labels = dataset[:,0]
	features = dataset[:,1:]	
	# take out polarization ID
	polarizationID = features[:,-1]
	features = np.delete(features, -1, 1)
	# take out weights
	weights = features[:,-1]
	features = np.delete(features, -1, 1)	
	
	return labels, features, weights, polarizationID

# trains model, returns test arrays and trained model
def train_model(test_index) :
	if test_index == 1:
		train_data = np.concatenate((set2, set3))
		ytest, Xtest, Wtest, Ptest = split_columns(set1)
	elif test_index == 2: 
		train_data = np.concatenate((set1, set3))
		ytest, Xtest, Wtest, Ptest = split_columns(set2)
	else :
		train_data = np.concatenate((set1, set2))
		ytest, Xtest, Wtest, Ptest = split_columns(set3)
	
	ytrain, Xtrain, Wtrain, Ptrain = split_columns(train_data)
	
	# reweight training events to avoid overfitting
	Wtrain = h.reweight_all(ytrain, Wtrain)
	
	xg_mod = xgb.XGBClassifier(objective = 'binary:logistic') 
	
	# use early stoppping rounds for training
	eval_set = [ (Xtest, ytest) ]
	xg_mod.fit(Xtrain, ytrain, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True, sample_weight=Wtrain)

	# return what is needed for analysis
	return xg_mod, Xtest, ytest, Wtest, Ptest


if __name__=="__main__":
		
	# import dataset (event order already randomized when creating the file)
	data = h.extract_data(file_name)

	# divide into  3 subsets of equal size
	set1, set2, set3 = np.array_split(data, 3, axis=0)
	
	# train 3 different times, each w/ a different test set 
	xg_mod1, Xtest1, ytest1, Wtest1, Ptest1 = train_model(1)
	xg_mod2, Xtest2, ytest2, Wtest2, Ptest2 = train_model(2)
	xg_mod3, Xtest3, ytest3, Wtest3, Ptest3 = train_model(3)
	
	#get predictions and probabilites from the three sets
	preds1 = xg_mod1.predict(Xtest1)
	preds2 = xg_mod2.predict(Xtest2)
	preds3 = xg_mod3.predict(Xtest3)
	proba1 = xg_mod1.predict_proba(Xtest1)
	proba1 = proba1[:,1]
	proba2 = xg_mod2.predict_proba(Xtest2)
	proba2 = proba2[:,1]
	proba3 = xg_mod3.predict_proba(Xtest3)
	proba3 = proba3[:,1]

	# SEPARATE PLOTS FOR EACH TRAIN/TEST ROUND
	"""
	h.plot_ROC_withProba(proba1, Xtest1, ytest1)
	h.plot_ROC_withProba(proba2, Xtest2, ytest2)
	h.plot_ROC_withProba(proba3, Xtest3, ytest3)

	h.output_threshold_plot_withProba(proba1, Xtest1, ytest1, True, Wtest1)
	h.output_threshold_plot_withProba(proba2, Xtest2, ytest2, True, Wtest2)
	h.output_threshold_plot_withProba(proba3, Xtest3, ytest3, True, Wtest3)	
	

	"""	
	# PLOTTING RESULTS TOGETHER (not sure if this is a good idea)
	
	# concatenate
	Xtest = np.concatenate((Xtest1, Xtest2))
	Xtest = np.concatenate((Xtest, Xtest3))
	ytest = np.concatenate((ytest1, ytest2))
	ytest = np.concatenate((ytest, ytest3))
	Wtest = np.concatenate((Wtest1, Wtest2))
	Wtest = np.concatenate((Wtest, Wtest3))
	Ptest = np.concatenate((Ptest1, Ptest2))
	Ptest = np.concatenate((Ptest, Ptest3))
	preds = np.concatenate((preds1, preds2))
	preds = np.concatenate((preds, preds3))
	proba = np.concatenate((proba1, proba2))
	proba = np.concatenate((proba, proba3))

	
	# percent accuract	
	print("Percent Accuracy: ", h.eval_preds(ytest,preds) )
		
	#plot ROC and calculate AUC
	h.plot_ROC_withProba(proba, Xtest, ytest)

	# probability distribution plot
	h.output_threshold_plot_withProba(proba, Xtest, ytest, True, Wtest, True, Ptest)
	Wtest = h.reweight_unity(Wtest, Ptest)
	h.output_threshold_plot_withProba(proba, Xtest, ytest, True,  Wtest, True, Ptest)
	
