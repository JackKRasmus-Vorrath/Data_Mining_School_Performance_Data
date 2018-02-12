#library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#to view visualizations in cell
get_ipython().magic('matplotlib inline')


#Read in raw data
schoolData = pd.read_csv('C:/Users/jkras/Desktop/All_Data_By_School_Final.csv',low_memory=False)
testScores = pd.read_csv('C:/Users/jkras/Desktop/1516testresults_masking_removed.csv', low_memory=False)
raceData = pd.read_csv('C:/Users/jkras/Desktop/Ec_Pupils_Expanded (2017 Race Compositions by School).csv',low_memory=False)


#Merging procedure below is same as it was in scripts for the previous lab

#pivot table for bringing in test score data
piv_test = pd.pivot_table(testScores, values='Percent GLP',index=['School Code'],columns='Subject')

#index used for merging
piv_test.index.name = 'unit_code'

#suffixing test score data with list comprehension
piv_test.columns = [str(col) + '_GLP' for col in piv_test.columns]

#resetting the index
piv_alltest = piv_test.reset_index()

#merging the main data set with the test score data
schoolData_alltest = schoolData.merge(piv_alltest,how='left',on='unit_code')


#list of feature names of interest from the race data
racecols = ['Indian Male', 'Indian Female', 'Asian Male',
       'Asian Female', 'Hispanic Male', 'Hispanic Female', 'Black Male',
       'Black Female', 'White Male', 'White Female', 'Pacific Island Male',
       'Pacific Island Female', 'Two or  More Male', 'Two or  More Female',
       'Total', 'White', 'Black', 'Hispanic', 'Indian', 'Asian',
       'Pacific Island', 'Two or More', 'White_Pct', 'Majority_Minority']

#suffixing the race data with list comprehension
racecols_renamed = [str(col) + '_RACE' for col in racecols]

#temporary dictionary for renaming
racecol_rename_dict = {i:j for i,j in zip(racecols,racecols_renamed)}

#rename features of race data
raceData.rename(index=str, columns=racecol_rename_dict,inplace=True)

#ensure that index of race data is of type string
raceData['unit_code'] = raceData['unit_code'].apply(str)

#fix the index of the race data to ensure it matches up with the indexes of the other two data sets
for i,j in raceData.iterrows():
    
    if len(raceData['unit_code'][i]) == 5:
        raceData.loc[i, 'unit_code'] = '0' + raceData['unit_code'][i]
        
#merging of test scores and school data with race data
schoolData_testRace = schoolData_alltest.merge(raceData,how='left',on='unit_code')



#bring in regional information from the test scores data set
region_df = testScores[['School Code','SBE District']]

#rename the index of the test score data to match that of the other data sets for merging
region_df.rename(index=str, columns={'School Code':'unit_code'},inplace=True)

#drop duplicate records
region_df_unique = region_df.drop_duplicates()

#final merging of the data sets
schoolData_testRaceRegion = schoolData_testRace.merge(region_df_unique,how='left',on='unit_code')

#check shape
schoolData_testRaceRegion.shape


#expenditures variable list
exp_list = ['lea_total_expense_num','lea_salary_expense_pct','lea_benefits_expense_pct','lea_services_expense_pct',
'lea_supplies_expense_pct','lea_instruct_equip_exp_pct']

#teacher qualifications variable list
teach_list = ['flicensed_teach_pct', 'tchyrs_0thru3_pct', 'tchyrs_4thru10_pct', 'tchyrs_11plus_pct', 'class_teach_num', 'nbpts_num', 'advance_dgr_pct',
'_1yr_tchr_trnovr_pct', 'emer_prov_teach_pct', 'lateral_teach_pct', 'highqual_class_pct', 'lea_flicensed_teach_pct',
'lea_tchyrs_0thru3_pct', 'lea_tchyrs_4thru10_pct', 'lea_tchyrs_11plus_pct', 'lea_class_teach_num', 'lea_nbpts_num', 'lea_advance_dgr_pct',
'lea_1yr_tchr_trnovr_pct', 'lea_emer_prov_teach_pct', 'lea_lateral_teach_pct', 'lea_highqual_class_pct', 'lea_highqual_class_hp_pct',
'lea_highqual_class_lp_pct', 'lea_highqual_class_all_pct', 'lea_not_highqual_class_hp_pct', 'lea_not_highqual_class_lp_pct',
'lea_not_highqual_class_all_pct']

#performance scores list
grade_list = ['SPG Score', 'Reading  SPG Score', 'Math SPG Score', 'EVAAS Growth Score', 'Overall Achievement Score', 'Read Score', 'Math Score',
'Science Score', 'Math I Score', 'English II Score', 'Biology Score', 'The ACT Score', 'ACT WorkKeys Score', 'Math Course Rigor Score',
'Cohort Graduation Rate Standard Score']




#alias full data set, merged from 3 CSV files
all_df = schoolData_testRaceRegion

#cast SBE district to categorical to generate dummy indicators, dropping reference level, and concatenating to full data set (original SBE District feature will be dropped below)
all_df['SBE District'] = all_df['SBE District'].astype('category')
Coded_District = pd.get_dummies(all_df['SBE District'],drop_first=True)
all_df = pd.concat([all_df,Coded_District],axis=1)


#scale the SPG Score feature
all_df['SPG Score Scaled'] = all_df['SPG Score']*(.01)
pd.Series.to_frame(all_df['SPG Score Scaled']).shape

#mean impute missing scaled SPG Score on region-wise basis; where no regional data, mean impute across all regions
mean_filled = all_df.groupby('SBE District')['SPG Score Scaled'].transform('mean')
all_df['SPG Score Scaled'] = all_df['SPG Score Scaled'].fillna(mean_filled)

all_df['SPG Score Scaled'] = pd.Series.to_frame(all_df['SPG Score Scaled']).fillna(pd.Series.to_frame(all_df['SPG Score Scaled']).mean())

all_df.shape


#column names from Race data, used for subsetting below
racecols = ['Indian Male', 'Indian Female', 'Asian Male',
       'Asian Female', 'Hispanic Male', 'Hispanic Female', 'Black Male',
       'Black Female', 'White Male', 'White Female', 'Pacific Island Male',
       'Pacific Island Female', 'Two or  More Male', 'Two or  More Female',
       'Total', 'White', 'Black', 'Hispanic', 'Indian', 'Asian',
       'Pacific Island', 'Two or More', 'White_Pct', 'Majority_Minority']

#suffixing Race columns for clarity
racecols_renamed = [str(col) + '_RACE' for col in racecols]


#creating full list of column names for subsetting below
logsvm_list_excludeDistrict = teach_list + exp_list + ['SPG Score Scaled'] + ['Northeast Region', 'Northwest Region', 'Piedmont Triad Region', 'SandHills Region', 'Southeast Region', 'Southwest Region', 'Western Region']


#NA values for the features below are arguably too numerous to mean impute
logsvm_list_excludeDistrict.remove('emer_prov_teach_pct')

logsvm_list_excludeDistrict.remove('lateral_teach_pct')

logsvm_list_excludeDistrict.remove('lea_emer_prov_teach_pct')

#for filtering the features of the all_df data frame below
logsvm_list_noDistrict_race = logsvm_list_excludeDistrict + racecols_renamed




#subset features of interest for mean imputation for-block below
all_df_nocharter = all_df[all_df['type_cd_txt'] != 'Charter']
logsvm_df = all_df_nocharter[logsvm_list_noDistrict_race]
logsvm_df.shape


#mean impute missing values on region-wise basis for all features of interest; where no regional data, mean impute across all regions
for col,name in zip(logsvm_df.T.values,logsvm_df.columns):
    if pd.isnull(col).any():
        mean_fill_col = all_df.groupby('SBE District')[name].transform('mean')
        all_df[name] = all_df[name].fillna(mean_fill_col)
        
        all_df[name] = pd.Series.to_frame(all_df[name]).fillna(pd.Series.to_frame(all_df[name]).mean())


#re-subset features of interest after mean imputation above

all_df_nocharter = all_df[all_df['type_cd_txt'] != 'Charter']
all_df_nocharter.shape
logsvm_df = all_df_nocharter[logsvm_list_noDistrict_race]
logsvm_df.shape



#remove 15 records without Race data
logsvm_df = logsvm_df[logsvm_df['Total_RACE'].notnull()]


#check null value percentages
#NA_report = logsvm_df.apply(lambda x: (sum(x.isnull().values)/len(x))*100, axis = 0)
#with pd.option_context('display.max_rows', None):
#    print("\n ****** Percentage of missing values in each attributes ********\n\n",NA_report)




#create categorization bins and names for scaled SPG Score
bins = [0.0,0.4,0.55,0.7,0.85,1.0]
group_names = ['F','D','C','B','A']

#create SPG Score Grade feature, based on binned SPG Score
logsvm_df['SPG Score_GRADE'] = pd.cut(logsvm_df['SPG Score Scaled'],bins,labels=group_names,include_lowest=True,right=False,precision=6)
#logsvm_df['SPG Score_GRADE'].tail(20)

#generate categorical codes: 0 = F, 1 = D, 2 = C, 3 = B, 4 = A
logsvm_df['SPG Score_GRADE'] = logsvm_df['SPG Score_GRADE'].cat.codes
#logsvm_df['SPG Score_GRADE'].tail(20)


#preparing dataframes for train-test-split to build models; SPG Score_GRADE used as classification target
#separate target variable and predictors
X = logsvm_df.drop('SPG Score_GRADE',axis=1)
X = X.drop('SPG Score Scaled',axis=1)
y = logsvm_df['SPG Score_GRADE']

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#train-test-split, 80% training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23233)


#multinomial logisitic regression 10-fold cross-validation model building
from sklearn.linear_model import LogisticRegressionCV
logmodel = LogisticRegressionCV(cv=10,solver='lbfgs',penalty='l2',refit=True,multi_class='multinomial')

#fit model on training data
logmodel.fit(X_train,y_train)

#generate predictions using test set predictors
predictions = logmodel.predict(X_test)

#model evaluation libraries
from sklearn.metrics import classification_report, confusion_matrix

#pass true test set values and predictions to classification_report
print(classification_report(y_test,predictions))

#generate confusion matrix for predictions
print(confusion_matrix(y_test,predictions))

#check accuracy score
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))




#SVM classifier hyperparameters identified using Grid Search Cross-Validation
from sklearn.grid_search import GridSearchCV

param_grid = {'C':[0.01,0.1,1,10,100],'gamma':[10,1,0.1,0.01,0.001]}

from sklearn.svm import SVR, SVC
grid = GridSearchCV(SVC(random_state=2323),param_grid,verbose=3)

#fit using training data and best hyperparameters
grid.fit(X_train,y_train)

#check most performant hyperparameters
grid.best_params_

#return best estimator signature
grid.best_estimator_

#return best score w/ best hyperparameters
grid.best_score_

#generate predictions using test set predictors
grid_predictions = grid.predict(X_test)

#generate confusion matrix and classification report using true test set values and predictions
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))



