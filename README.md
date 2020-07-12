# xgboost-analysis

Repository of XGBoost Analysis Tools used for Classification of Longitudinal-Longitudinal WZ Boson events

process_GEANT.py : 
  - required input files: LL, LT, TL, TT output files from GEANT simulation
  - parses through nominal tree to extract event information into a pandas dataframe (method from: https://gitlab.cern.ch/shuzhou/mva-in-bbll/-/blob/master/python/bbll/preprocess.py)
  - three options for selection cuts: no cuts, lepton cuts only, all cuts except costhetaW

split.py :
  - requires input file of data w/ columns in this order: label features weight polarizationID (can be prepared with process_GEANT.py)
  - splits into Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest, Ptest and saves to .csv files

xgb_weighted.py :
  - requires input files: Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest, Ptest (can be prepared using split.py)
  - trains xgboost model and tests
  - outputs feature importance bar chart, ROC plot w/ AUC score, normalized probability distribution plot, shape-comparison probability distribution plot

xgb_unweighted.py :
  - same as xgb_weighted.py, but for datasets w/out weighting

xgb_tuning.py :
  - based off of this tutorial: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
  - requires input files: Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest (can be prepared using split.py), and original input file of data (prepared using process_GEANT.py)
  - uses grid search / cross validation to tune hyperparameters of XGBoost 
  - plots comparison of ROC curve before and after hyperparameter tuning

xgb_crossval.py :
  - requires an input file of data w/ columns in this order: label features weight polarizationID (can be prepared with process_GEANT.py)
  - performs 3-fold cross validation with reweighting of training events & early stopping rounds implemented to avoid overfitting
  - two options: concatenate 3 distinct test sets / output probability sets to create one set of ROC/probability distribution plots
                 plot three separate ROC/probability distribution plots, one for each round of train/test
           
helpers.py :
  - helper functions used across all files kept here
