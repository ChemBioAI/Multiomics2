
#calculate mean and std for normal vs cancer 
#welch's t - test 

# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#load dataset after transforming the dataset in the range of 0-1
data_train=pd.read_csv('merged_final.csv')
data_train.set_index(data_train.columns[0], inplace=True)





#Cancer with 0 and Normal with 1
data_tumour = data_train[data_train.Condition == 0]
data_normal =  data_train[data_train.Condition == 1]

data_normal
#Caculate mean and std 

data_means_normal = data_normal.iloc[:,:-1].mean()
data_means_normal = pd.DataFrame(data_means_normal)
# data_means.columns = ['normal_mean'] do this when we have the remaining means for the data 
data_means_normal = data_means_normal

data_means_normal


data_std_normal = data_normal.iloc[:,:-1].std()
data_std_normal = pd.DataFrame(data_std_normal)
# data_means.columns = ['normal_mean'] do this when we have the remaining means for the data 
data_std_normal = data_std_normal

data_means_tumour = data_tumour.iloc[:,:-1].mean()
data_means_tumour = pd.DataFrame(data_means_tumour)
# data_means.columns = ['normal_mean'] do this when we have the remaining means for the data 
data_means_tumour = data_means_tumour

data_std_tumour = data_tumour.iloc[:,:-1].std()
data_std_tumour = pd.DataFrame(data_std_tumour)
# data_means.columns = ['normal_mean'] do this when we have the remaining means for the data 
data_std_tumour = data_std_tumour


data_means = pd.concat([data_means_normal,data_std_normal,data_means_tumour,data_std_tumour] , axis = 1 , ignore_index= True )
data_means.columns = ['Normal_mean','Normal_std','Tumour_mean','Tumour_std']
data_means

features = data_means.index
features


#####################################################################################################################################
""" 
Performing welch's t - test 
"""
####################################################################################################################################

from scipy.stats import ttest_ind_from_stats
P_EL = []
for x in range(0,len(data_means)):
  a = ttest_ind_from_stats(mean1 = data_means['Normal_mean'][x], std1 = data_means['Normal_std'][x] ,nobs1 = len(data_normal) , mean2 = data_means['Tumour_mean'][x] , std2 =  data_means['Tumour_std'][x] , nobs2= len(data_tumour), equal_var = False )
  P_EL.append(a[1])


#####################################################################################################################################
""" 
false discovery rate (FDR) correction
"""
####################################################################################################################################
#import pandas.util.testing as tm
from statsmodels.stats.multitest import fdrcorrection
p_el_mt = fdrcorrection(P_EL , alpha = 0.05 , method = 'indep' , is_sorted = False)

p_el_mt

elec_feat = features[p_el_mt[0]]

elec_feat


ttesttable = pd.concat([pd.DataFrame(features), pd.DataFrame(P_EL)] , axis = 1 )
ttesttable = pd.concat([pd.DataFrame(ttesttable), pd.DataFrame(p_el_mt[1])] , axis = 1 )
ttesttable.columns = ['features', 'p_value' , 'fdr_adj_value']
ttesttable


ttesttable.to_csv('ttesttable.csv')

data_means['log2(a/b)']= np.log2(data_means['Tumour_mean']/data_means['Normal_mean'])
print(data_means)
data_means.to_csv('data_means_final.csv')


