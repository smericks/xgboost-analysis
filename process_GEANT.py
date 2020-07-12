# adapted from: https://gitlab.cern.ch/shuzhou/mva-in-bbll/-/blob/master/python/bbll/preprocess.py
# for this dataset, use n_tumple_name: "nominal"
from ROOT import TMVA, TFile, TTree, TCut, TString, TH1F, TCanvas
import numpy as np 
import pandas as pd

variable_list = { "Yields", "isMC", "Channel", "Year", "NormSF", "WeightSign", "WeightNormalized", "Weight", "M_WZ", "Mt_WZ", "M_Z", "Mt_W", "Met", "Njets", "NBjets", "Lep1Pt", "Lep2Pt", "Lep3Pt", "Lep1Eta", "Lep2Eta", "Lep3Eta", "Lep1Phi", "Lep2Phi", "Lep3Phi", "Mpx", "Mpy", "Mpz", "Lep1WeightZ", "Lep2WeightZ", "Lep3WeightZ", "Lep1WeightW", "Lep2WeightW", "Lep3WeightW", "Pt_W", "Pt_Z", "Pt_WZ", "CosThetaV", "CosThetaLepW", "CosThetaLepZ", "DY_WZ", "DY_3Z", "DY_3N" }

LL_nam = "polarization.364991_MGPy8EG_WZ_CKKWL_lvll_LO_WLZLPol_ntuples.root"
LT_nam = "polarization.364992_MGPy8EG_WZ_CKKWL_lvll_LO_WLZTPol_ntuples.root"
TL_nam = "polarization.364993_MGPy8EG_WZ_CKKWL_lvll_LO_WTZLPol_ntuples.root"
TT_nam = "polarization.364994_MGPy8EG_WZ_CKKWL_lvll_LO_WTZTPol_ntuples.root"	

def make_lepton_cuts(pd_temp) :
	# input/output: pandas dataframe
	# removes events from pd_temp that do not pass lepton cuts
	v = pd_temp[ ['Lep1Pt', 'Lep2Pt', 'Lep3Pt', 'Lep1Eta', 'Lep2Eta', 'Lep3Eta', 'Met'] ]
	
	arr = v.values
	for i in range(0, len(v.index)) :
		if (arr[i][0] < 25 or arr[i][1] < 25 or arr[i][2] < 25 ) :
			pd_temp = pd_temp.drop([i], axis=0)
		elif ( abs(arr[i][3]) > 2.5 or abs(arr[i][4]) > 2.5 or abs(arr[i][5]) > 2.5 ) :
			pd_temp = pd_temp.drop([i], axis=0)
		elif ( arr[i][6] < 25 ) :
			pd_temp = pd_temp.drop([i], axis=0)
	return pd_temp

def make_cuts(pd_temp) :
	# input/output: pandas dataframe
	# makes lepton cuts, pT Z cut, pT WZ cut
	# removes events from pd_temp that do not pass the cuts
	v = pd_temp[ ['Lep1Pt', 'Lep2Pt', 'Lep3Pt', 'Lep1Eta', 'Lep2Eta', 'Lep3Eta', 'Met', 'Pt_Z', 'Pt_WZ'] ]
	
	arr = v.values	
	for i in range(0, len(v.index) ) :
		# lepton cuts
		if ( arr[i][0] < 25 or arr[i][1] < 25 or arr[i][2] < 25 ) :
			pd_temp = pd_temp.drop([i], axis=0)
		elif ( abs(arr[i][3]) > 2.5 or abs(arr[i][4]) > 2.5 or abs(arr[i][5]) > 2.5 ) :
			pd_temp = pd_temp.drop([i], axis=0)
		elif ( arr[i][6] < 25 ) :
			pd_temp = pd_temp.drop([i], axis=0)
		# pT Z cut
		elif ( arr[i][7] < 200 ) :
			pd_temp = pd_temp.drop([i], axis=0)
		# pT WZ cut
		elif ( arr[i][8] > 70 ) :
			pd_temp = pd_temp.drop([i], axis=0)
	return pd_temp	

def make_dataFrame(filename, isLL) :
	File_temp = TFile.Open(filename)	
	ntuple_temp = File_temp.Get("nominal")
	Array_temp = ntuple_temp.AsMatrix(variable_list)
	pd_temp = pd.DataFrame( data = Array_temp, columns = variable_list)
	# perform selection cuts
	pd_temp = make_cuts(pd_temp)
	pd_temp = pd_temp.sort_index(axis=1)
	sz = len(pd_temp.index) # returns number of rows
	if isLL == True :
		l = np.ones(sz)
	else :
		l = np.zeros(sz)
	# add signal/background label
	pd_temp['is_signal'] = l
	p = pd_temp[ ['is_signal', 'CosThetaV', 'CosThetaLepW', 'CosThetaLepZ','DY_WZ', 'DY_3N', 'DY_3Z', 'WeightNormalized'] ]
	return p


print("LL")
LL = make_dataFrame(LL_nam, True)
print("LT")
LT = make_dataFrame(LT_nam, False)
print("TL")
TL = make_dataFrame(TL_nam, False)
print("TT")
TT = make_dataFrame(TT_nam, False)

# add polarization ID at the end of each row, used later for probability distribution plotting
ll = np.append(LL.values, np.zeros([len(LL.values),1]),1)
lt = np.append(LT.values, np.ones([len(LT.values),1]),1)
tl = np.append(TL.values, 2*np.ones([len(TL.values),1]),1)
tt = np.append(TT.values, 3*np.ones([len(TT.values),1]),1) 

events = np.concatenate((ll, lt))
events = np.concatenate((events, tl))
events = np.concatenate((events, tt))
np.random.shuffle(events)
print(events)
np.savetxt("GEANT_polar_allcuts.txt", events)

