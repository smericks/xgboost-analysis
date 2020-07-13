import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc



def reweight_TT(ytrain, Wtrain) :
        # reweights training events s.t. signal and background have equal contribution
        # assumes a TT-only background
	signal, background = 0,0
	for i in ytrain :
                if i == 0 :
                        background = background + 1
                if i ==1 :
                        signal = signal + 1

	print("Background Count: ", background)
	print("Signal Count: ", signal)
	scale = background / signal
	Wtrain[ytrain == 0] = 1
	Wtrain[ytrain == 1] = scale
	return Wtrain

def reweight_all(ytrain, Wtrain) :
	# reweights training events s.t. signal and background have equal contribution
	# assumes a LT, TL, TT combined background
	sz = len(ytrain)
	background, signal_count = 0,0
	for i in range(0,sz) :
		if ytrain[i] == 1 :
			signal_count = signal_count + 1					
		if ytrain[i] ==0 :	
			background = background + Wtrain[i]
	scale = background / signal_count
	Wtrain[ytrain == 1] = scale
	return Wtrain

def reweight_unity(Wtest, Ptest) :
	# used for shape comparison plot
	# takes in array of test weights, test polarization ID (0=LL, 1=LT, 2=TL, 3=TT) 
	# re-weights test events s.t. all polarizations have equal contribution
	LL, LT, TL, TT = 0,0,0,0
	for p in Ptest :
		if p == 0 :
			LL = LL + 1
		elif p == 1 :
			LT = LT + 1
		elif p == 2 :
			TL = TL + 1
		else :
			TT = TT + 1
	
	Wtest[Ptest==0] = TT / LL
	Wtest[Ptest==1] = TT / LT
	Wtest[Ptest==2] = TT / TL
	Wtest[Ptest==3] = 1
	
	return Wtest
	
def extract_data(file_name, remove_first=False) :
	# simply returns contents of a txt file as an array
	# if remove_first = True, first line is skipped 
	data = []
	file = open(file_name)
	count = 0
	for line in file :
		if remove_first and ( abs(count-0) < 0.01 ) :
			count = 3
		else :	
			event = line.split(" ")
			event = np.array(event)
			event = event.astype(np.float)
			data.append(event)
	return data

def signal_fraction(file_name) :
	# computes signal fraction of a dataset
	data = extract_data(file_name)
	data = np.array(data)
	labels = data[:,0]
	signal, background = 0, 0
	for l in labels :
		if l == 1:
			signal = signal + 1
		else :
			background = background + 1
	print("Signal Fraction: ", signal / (signal + background) )
		


def extract_split(file_name, testsize, seed) :
	#assumes labels first
	#returns Xtrain, Xtest, ytrain, ytest 
	dataset = extract_data(file_name)
	dataset = np.array(dataset)
	labels = dataset[:,0]
	features = dataset[:,1:]

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=testsize, random_state=seed)
	return X_train, X_test, y_train, y_test

def extract_split_weights(file_name, testsize, seed) :
	#assumes labels first, weights last
	# returns Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest
	dataset = extract_data(file_name)
	dataset = np.array(dataset)
	labels = dataset[:,0]
	features = dataset[:,1:]
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=testsize, random_state=seed)
	# now, take out weights from X_train, X_test
	train_weights = X_train[:,-1]
	test_weights = X_test[:,-1]
	X_train = np.delete(X_train, -1, 1)
	X_test = np.delete(X_test, -1, 1)

	return X_train, X_test, y_train, y_test, train_weights, test_weights

def extract_split_polarizations(file_name, testsize, seed, if_nofile=False, data_arr=0) :
        #assumes labels first, weights second-to-last, polarization ID last
	# returns Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest, Ptest (polarization ID for test events)
	# if you already have an array w/ your dataset, use if_nofile option
	if if_nofile == False :
		dataset = extract_data(file_name)
		dataset = np.array(dataset)
	else :
		dataset = data_arr
	
	labels = dataset[:,0]
	features = dataset[:,1:]
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=testsize, random_state=seed)	
	# now, take out polarizations from X_train, X_test
	P_train = X_train[:,-1]
	P_test = X_test[:,-1]
	X_train = np.delete(X_train, -1, 1)
	X_test = np.delete(X_test, -1, 1)
	# next, take out weights from X_train, X_test
	train_weights = X_train[:,-1]
	test_weights = X_test[:,-1]
	X_train = np.delete(X_train, -1, 1)
	X_test = np.delete(X_test, -1, 1)	

	return X_train, X_test, y_train, y_test, train_weights, test_weights, P_test

def eval_preds(y, preds) :
	# evaluate predictions based off of % accuracy
	# returns percent accuracy
	num_corr = 0
	num_wrong = 0
	s = len(y)
	n = np.arange(0,s,1)
	for i in n :
		if abs(y[i] - preds[i]) < 0.01 :
			num_corr = num_corr + 1
		else :
			num_wrong = num_wrong + 1
	perc_acc = (num_corr / s) * 100
	# print("Number Correct: ", num_corr)
	# print("Number Incorrect: ", num_wrong)
	# print("Percent Accuracy: ", perc_acc)
	return perc_acc

def plot_trees(model,num_trees) : 
	#plots all trees and saves them each to a .png file
	trees = np.arange(0,num_trees,1)
	for i in trees :
		xgb.plot_tree(model, num_trees=i)
		fig = plt.gcf()
		fig.set_size_inches(150,100)
		num = str(i)
		name = "tree_" + num + ".png"
		fig.savefig(name)

def plot_importance(model) :
	# plots a bar graph of feature importance
	xgb.plot_importance(model)
	plt.rcParams['figure.figsize'] = [5,5]
	plt.show()

def plot_ROC_withProba(preds, Xtest, ytest, DontShow=False) :
	fpr, tpr, _ = roc_curve(ytest, preds)
	auc_score = auc(fpr, tpr)
	print("AUC Score: ", auc_score)
	plt.clf()
	plt.title('ROC Curve')
	plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
	plt.plot([0,1],[0,1],'r--')
	
	threshold, distance, f, t = best_threshold_withProba(preds, Xtest, ytest)
	plt.plot(f, t, 'ro', label='Best Threshold = {:.3f}'.format(threshold))
	
	plt.xlim([-0.1, 1.1])
	plt.ylim([-0.1, 1.1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')

	if DontShow == False :
		plt.legend(loc='lower right')
		plt.show() 

def plot_ROC(xg_mod, Xtest, ytest, comparison=False, Xtrain=0, ytrain=0) :
	# plots an ROC curve and computes AUC score
	# if comparison is true, needs Xtrain and ytrain as inputs to plot default ROC curve
	y_preds = xg_mod.predict_proba(Xtest)
	preds = y_preds[:,1]

	plot_ROC_withProba(preds, Xtest, ytest, True)
		
	if comparison == True :
		mod = xgb.XGBClassifier(objective = 'binary:logistic')
		mod.fit(Xtrain,ytrain)
		y_preds2 = mod.predict_proba(Xtest)
		preds2 = y_preds2[:,1]
		fpr2, tpr2, _ = roc_curve(ytest, preds2)
		auc_score2 = auc(fpr2, tpr2)
		preds_acc = mod.predict(Xtest)
		accuracy = eval_preds(ytest,preds_acc)
		print("Default % accuray: ", accuracy)
		print("Default AUC Score: ", auc_score2)
		plt.plot(fpr2, tpr2, label='Default AUC = {:.3f}'.format(auc_score2))
			
	plt.legend(loc='lower right')
	plt.show()

def output_threshold_plot_withProba(preds, Xtest, ytest, is_weighted=False,  Wtest=0, polarized=False, Ptest=0) :
	# creates a plot of probability distribution for signal & background using histograms
	# if is_weighted, will weight events using Wtest
	# if polarized, will plot 4 separate histograms (one for each polarization)
	plt.figure(figsize=(15,7) )
	
	if polarized :
		plt.hist(preds[Ptest==0], weights=Wtest[Ptest==0], bins=15, label='LL', histtype=u'step')
		plt.hist(preds[Ptest==1], weights=Wtest[Ptest==1], bins=15, label='LT', histtype=u'step')
		plt.hist(preds[Ptest==2], weights=Wtest[Ptest==2], bins=15, label='TL', histtype=u'step')
		plt.hist(preds[Ptest==3], weights=Wtest[Ptest==3], bins=15, label='TT', histtype=u'step')
	if is_weighted and polarized==False :
		plt.hist(preds[ytest==0], weights=Wtest[ytest==0], bins=15, label='Background', histtype=u'step')
		plt.hist(preds[ytest==1], weights=Wtest[ytest==1], bins=15, label='Signal', histtype=u'step')	
	if is_weighted==False :
		plt.hist(preds[ytest==0], bins=15, label='Background', histtype=u'step')
		plt.hist(preds[ytest==1], bins=15, label='Signal', histtype=u'step')
	
	threshold, _, _, _ = best_threshold_withProba(preds, Xtest, ytest)
	#plt.axvline(x=threshold, color = 'r', linestyle='dashed', label='Best Threshold')
	plt.xlabel('Probability of being Signal event')
	plt.ylabel('Num events')
	plt.legend(fontsize=20)
	#plt.yticks(range(0,11,2))
	plt.tick_params(axis='both', labelsize=25, pad=5)
	plt.show()

	sz = len(preds)
	signal = 0
	total_events = 0
	for i in range(0,sz) :
		if preds[i] > threshold :
			signal = signal + ytest[i] # will add 0 for background events
			total_events = total_events + 1
	LL_fraction = signal / total_events
	print("LL fraction after thresholding: ", LL_fraction)

def output_threshold_plot(xg_mod, Xtest, ytest, is_weighted=False, Wtest=0, polarized=False, Ptest=0) :
	preds = xg_mod.pred_proba(Xtest)
	preds = preds[:,1]
	output_threshold_plot_withProba(preds, Xtest, ytest, is_weighted, Wtest, polarized, Ptest)

def best_threshold_withProba(preds, Xtest, ytest) :
	fpr, tpr, thresholds = roc_curve(ytest, preds)
	sz = len(fpr)
	min_distance = 1
	min_index = 0
	for i in range(0,sz) :
		distance = np.sqrt( fpr[i]*fpr[i] + (1 - tpr[i])*(1 - tpr[i]) )
		if distance < min_distance :
			min_distance = distance
			min_index = i
	print("Best Threshold: ", thresholds[min_index])
	print("Distnace from Ideal: ", min_distance)
	print("FPR: ", fpr[min_index])
	print("TPR: ", tpr[min_index])
	
	return thresholds[min_index], min_distance, fpr[min_index], tpr[min_index]

def best_threshold(xg_mod, Xtest, ytest) :
	# finds threshold that is closest to fpr=0, tpr=1 (ideal point on ROC curve)
	# returns best threshold, distance to ideal point of that threshold, fpr, tpr  
	preds = xg_mod.predict_proba(Xtest)
	preds = preds[:,1]
	a, b, c, d = best_threshold_withProba(preds, Xtest, ytest)
	return a, b, c, d
