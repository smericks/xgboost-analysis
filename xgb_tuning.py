# adapted from: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import helpers as h
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc

np.set_printoptions(threshold=np.inf)
# SET IF WEIGHTED HERE
weighted = True

Xtrain = np.loadtxt('Xtrain_truthTT.csv', delimiter=',')
Xtest = np.loadtxt('Xtest_truthTT.csv', delimiter=',')
ytrain = np.loadtxt('ytrain_truthTT.csv', delimiter=',')
ytest = np.loadtxt('ytest_truthTT.csv', delimiter=',')
Wtrain = np.loadtxt('Wtrain_truthTT.csv', delimiter=',')
# reweight training events
Wtrain = h.reweight(ytrain, Wtrain)
Wtest = np.loadtxt('Wtest_truthTT.csv', delimiter=',')
dataset = np.array(h.extract_data("truth_onlyTT.txt"))
print(dataset)
#assumes labels are first, features are second
labels = dataset[:,0]
#delete labels
features = np.delete(dataset, [0], 1)
#only if using weights
if weighted :
	weights = features[:,-1]
	# reweight for training
	weights = h.reweight(labels, weights)
	features = np.delete(features, -1, 1)

def tune_learningrate(max_depth, min_child_weight, gamma, subsample, colsample_bytree) :
	#STEP 1: tune learning rate & num trees at the same time
	trees = tune_numtrees(max_depth, min_child_weight, gamma, subsample, colsample_bytree, 0.1, 3)
	opt_trees = trees
	opt_rate = 0.1
	if trees > 100 : 
		trees2 = tune_numtrees(max_depth, min_child_weight, gamma, subsample, colsample_bytree, 0.2, 3)
		opt_trees, opt_rate = trees2, 0.2	
		if trees2 > 100 : 
			trees3 = tune_numtrees(max_depth, min_child_weight, gamma, subsample, colsample_bytree, 0.3, 3)
			opt_trees, opt_rate = trees3, 0.3
	return opt_rate, opt_trees

def tune_numtrees(max_depth, min_child_weight, gamma, subsample, colsample_bytree, rate, nfolds) :
	#Find optimal number of trees (estimators) using xgboost.cv()
	if weighted :
		data_dmatrix = xgb.DMatrix(data=features, label=labels, weight=weights)
	else :
		data_dmatrix = xgb.DMatrix(data=features,label=labels)
	params = {"objective":"binary:logistic",'learning_rate':rate,'max_depth':max_depth,'gamma':gamma,'subsample':subsample, 'colsample_bytree':colsample_bytree}
	cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=nfolds, metrics='auc', num_boost_round=500, early_stopping_rounds=10,seed=123)

	estop = 10
	best = (cv_results.shape[0] - estop) / (1 - 1/nfolds) 
	print('learning rate: ', rate)
	print('best num rounds: ', best)
	best = np.floor(best)
	return int(best)

def tune_step2(learning_rate, n_estimators, gamma, subsample, colsample_bytree) :
	#STEP 2: Tune max_depth and min_child_weight
	# Tuned 1st b/c they have highest impact on model output
	# returns optimal depth, optimal min child weight
	output = []
	depths = np.arange(3,10,1)
	child_weights = np.arange(1,6,1)
	for d in depths :
		for c in child_weights :
			#define model
			xg_mod = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = learning_rate, n_estimators=n_estimators, max_depth = d, min_child_weight=c, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree)
			#train & test
			if weighted :
				xg_mod.fit(Xtrain, ytrain, sample_weight=Wtrain)
			else :
				xg_mod.fit(Xtrain,ytrain)
			print("Done with d: ", d, " c: ", c)
			preds = xg_mod.predict(Xtest)
			accuracy = h.eval_preds(ytest,preds)
			y_preds = xg_mod.predict_proba(Xtest)
			preds1 = y_preds[:,1]
			fpr, tpr, _ = roc_curve(ytest, preds1)
			auc_score = auc(fpr,tpr)	
			output.append([d, c, accuracy, auc_score])
	o = max_output(output, 2, 3)
	return o[0], o[1]
 
def max_output(output, acc_i, auc_i) : 
	#find max output
	#assumes 3rd index is % accuracy, 4th index is AUC
	#returns row of maximum AUC 
	sz = len(output)
	s = np.arange(0,sz,1)
	max_acc, max_idx = 0,0
	max_auc, max_idx2 = 0,0
	for i in s :
		if output[i][acc_i] > max_acc :
			max_acc = output[i][acc_i]
			max_idx = i
		if output[i][auc_i] > max_auc : 
			max_auc = output[i][auc_i]
			max_idx2 = i

	print("Max Accuracy: ")
	print(output[max_idx])
	print("Max AUC: ")
	print(output[max_idx2])
	return output[max_idx2]

def tune_gamma(rate, num_trees, max_depth, child_weight, subsample, colsample_bytree) :
	#STEP 3: Tune Gamma
	#returns optimal gamma value
	gammas = np.arange(0,0.6,0.1)
	output = []
	for g in gammas : 
		#define model
		xg_mod = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = rate, n_estimators=num_trees,  max_depth = max_depth, min_child_weight=child_weight, gamma=g, subsample=subsample, colsample_bytree=colsample_bytree)
                #train & test
		if weighted :
			xg_mod.fit(Xtrain, ytrain, sample_weight = Wtrain)
		else :
			xg_mod.fit(Xtrain,ytrain)
		preds = xg_mod.predict(Xtest)
		accuracy = h.eval_preds(ytest,preds)
		y_preds = xg_mod.predict_proba(Xtest)
		preds1 = y_preds[:,1]
		fpr, tpr, _ = roc_curve(ytest, preds1)
		auc_score = auc(fpr,tpr)
		output.append([g, accuracy, auc_score])
	o = max_output(output, 1, 2)
	return o[0]

def tune_step5(rate, num_trees, max_depth, child_weight, g) :
	#STEP 5: Tune subsample and colsample_bytree
	#returns optimal subsample and colsample_bytree
	subsamples = np.arange(0.6,0.95,0.05)
	colsamples = np.arange(0.6,0.95,0.05)
	output = []
	for s in subsamples :
		for c in colsamples :
        		#define model
			xg_mod = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = rate, n_estimators=num_trees,  max_depth = max_depth, min_child_weight=child_weight, gamma=g, subsample=s, colsample_bytree=c)
                	#train & test
			if weighted :
				xg_mod.fit(Xtrain, ytrain, sample_weight = Wtrain)
			else :
				xg_mod.fit(Xtrain,ytrain)
			preds = xg_mod.predict(Xtest)
			accuracy = h.eval_preds(ytest,preds)
			y_preds = xg_mod.predict_proba(Xtest)
			preds1 = y_preds[:,1]
			fpr, tpr, _ = roc_curve(ytest, preds1)
			auc_score = auc(fpr,tpr)
			output.append([s, c, accuracy, auc_score])
	o = max_output(output, 2, 3)
	return o[0], o[1]


def final_model(rate, num_trees, depth, child_weight, g, s, c) :
	#STEP 6: Plot ROC and find AUC of final modeli
	print('Final Model Parameters: learning rate = ', rate, ', num trees = ', num_trees, ', max depth = ', depth, ', min child weight = ', child_weight, ', gamma = ', g, ', subsample = ', s, ', columns sampled per tree = ', c) 
	xg_mod = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = rate, n_estimators=num_trees, max_depth = depth, min_child_weight=child_weight, gamma=g, subsample=s, colsample_bytree=c)
	#train & test
	if weighted :
		xg_mod.fit(Xtrain, ytrain, sample_weight= Wtrain)
		# early stopping version
		#eval_set = [ (Xtest, ytest) ]
		#xg_mod.fit(Xtrain, ytrain, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True, sample_weight=Wtrain)
		h.output_threshold_plot(xg_mod, Xtest, ytest, True, Wtest)
	else :
		xg_mod.fit(Xtrain,ytrain)
		h.output_threshold_plot(xg_mod, Xtest, ytest)
	
	h.plot_ROC(xg_mod, Xtest, ytest, True,  Xtrain, ytrain)

if __name__=="__main__":
	learning_rate, num_trees = tune_learningrate(5, 1, 0, 0.8, 0.8)
	print("Step 1 complete")
	max_depth, min_child_weight = tune_step2(learning_rate, num_trees, 0, 0.8, 0.8)
	print("Step 2 complete")
	g = tune_gamma(learning_rate, num_trees, max_depth, min_child_weight, 0.8, 0.8) 
	print("Step 3 complete")
	# re-calibrate number of trees
	num_trees = tune_numtrees(max_depth, min_child_weight, g, 0.8, 0.8, learning_rate, 3) 	
	print("Step 4 complete")
	subsample, colsample = tune_step5(learning_rate, num_trees, max_depth, min_child_weight, g)
	print("Step 5 complete, analyzing final model ... ")
	# print out final model parameters & plot ROC curve w/ AUC
	final_model(learning_rate, num_trees, max_depth, min_child_weight, g, subsample, colsample)
