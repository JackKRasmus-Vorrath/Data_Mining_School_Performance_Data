#GRADIENT BOOSTING REGRESSION & DENSE NEURAL NETWORK CLASSIFICATION

#Same import and cleaning procedure as in previous labs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

#Read in raw data
schoolData = pd.read_csv('C:/Users/jkras/Desktop/All_Data_By_School_Final.csv',low_memory=False)
testScores = pd.read_csv('C:/Users/jkras/Desktop/1516testresults_masking_removed.csv', low_memory=False)
raceData = pd.read_csv('C:/Users/jkras/Desktop/Ec_Pupils_Expanded (2017 Race Compositions by School).csv',low_memory=False)

#gotta merge 'em all!
piv_test = pd.pivot_table(testScores, values='Percent GLP',index=['School Code'],columns='Subject')

piv_test.index.name = 'unit_code'

piv_test.columns = [str(col) + '_GLP' for col in piv_test.columns]

piv_alltest = piv_test.reset_index()

schoolData_alltest = schoolData.merge(piv_alltest,how='left',on='unit_code')

racecols = ['Indian Male', 'Indian Female', 'Asian Male',
       'Asian Female', 'Hispanic Male', 'Hispanic Female', 'Black Male',
       'Black Female', 'White Male', 'White Female', 'Pacific Island Male',
       'Pacific Island Female', 'Two or  More Male', 'Two or  More Female',
       'Total', 'White', 'Black', 'Hispanic', 'Indian', 'Asian',
       'Pacific Island', 'Two or More', 'White_Pct', 'Majority_Minority']

racecols_renamed = [str(col) + '_RACE' for col in racecols]

racecol_rename_dict = {i:j for i,j in zip(racecols,racecols_renamed)}

raceData.rename(index=str, columns=racecol_rename_dict,inplace=True)

raceData['unit_code'] = raceData['unit_code'].apply(str)

for i,j in raceData.iterrows():
    
    if len(raceData['unit_code'][i]) == 5:
        raceData.loc[i, 'unit_code'] = '0' + raceData['unit_code'][i]
        
schoolData_testRace = schoolData_alltest.merge(raceData,how='left',on='unit_code')

region_df = testScores[['School Code','SBE District']]

region_df.rename(index=str, columns={'School Code':'unit_code'},inplace=True)

region_df_unique = region_df.drop_duplicates()

schoolData_testRaceRegion = schoolData_testRace.merge(region_df_unique,how='left',on='unit_code')

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

#create percentages
logsvm_df['Indian_RACE_Pct'] = logsvm_df['Indian_RACE']/logsvm_df['Total_RACE']
logsvm_df['Asian_RACE_Pct'] = logsvm_df['Asian_RACE']/logsvm_df['Total_RACE']
logsvm_df['Hispanic_RACE_Pct'] = logsvm_df['Hispanic_RACE']/logsvm_df['Total_RACE']
logsvm_df['Black_RACE_Pct'] = logsvm_df['Black_RACE']/logsvm_df['Total_RACE']
logsvm_df['Pacific Island_RACE_Pct'] = logsvm_df['Pacific Island_RACE']/logsvm_df['Total_RACE']
logsvm_df['Two or More_RACE_Pct'] = logsvm_df['Two or More_RACE']/logsvm_df['Total_RACE']

#masking variables to avoid division-by-zero errors
mask_Indian = (logsvm_df['Indian Male_RACE'] == 0) & (logsvm_df['Indian_RACE'] == 0)
mask_Asian = (logsvm_df['Asian Male_RACE'] == 0) & (logsvm_df['Asian_RACE'] == 0)
mask_Hispanic = (logsvm_df['Hispanic Male_RACE'] == 0) & (logsvm_df['Hispanic_RACE'] == 0)
mask_Black = (logsvm_df['Black Male_RACE'] == 0) & (logsvm_df['Black_RACE'] == 0)
mask_White = (logsvm_df['White Male_RACE'] == 0) & (logsvm_df['White_RACE'] == 0)
mask_Pacific = (logsvm_df['Pacific Island Male_RACE'] == 0) & (logsvm_df['Pacific Island_RACE'] == 0)
mask_TwoRace = (logsvm_df['Two or  More Male_RACE'] == 0) & (logsvm_df['Two or More_RACE'] == 0)

#create gender percentages
logsvm_df['Indian_RACE_Male_Pct'] = logsvm_df['Indian Male_RACE'].div(logsvm_df['Indian_RACE']).where(~mask_Indian,0)
logsvm_df['Asian_RACE_Male_Pct'] = logsvm_df['Asian Male_RACE'].div(logsvm_df['Asian_RACE']).where(~mask_Asian,0)
logsvm_df['Hispanic_RACE_Male_Pct'] = logsvm_df['Hispanic Male_RACE'].div(logsvm_df['Hispanic_RACE']).where(~mask_Hispanic,0)
logsvm_df['Black_RACE_Male_Pct'] = logsvm_df['Black Male_RACE'].div(logsvm_df['Black_RACE']).where(~mask_Black,0)
logsvm_df['White_RACE_Male_Pct'] = logsvm_df['White Male_RACE'].div(logsvm_df['White_RACE']).where(~mask_White,0)
logsvm_df['Pacific_Island_RACE_Male_Pct'] = logsvm_df['Pacific Island Male_RACE'].div(logsvm_df['Pacific Island_RACE']).where(~mask_Pacific,0)
logsvm_df['Two_or_More_RACE_Male_Pct'] = logsvm_df['Two or  More Male_RACE'].div(logsvm_df['Two or More_RACE']).where(~mask_TwoRace,0)

#drop multicolinear features
logsvm_df.drop(['Indian_RACE','Asian_RACE','Hispanic_RACE','Black_RACE','White_RACE','Pacific Island_RACE','Two or More_RACE',
                       'Indian Male_RACE','Asian Male_RACE','Hispanic Male_RACE','Black Male_RACE','White Male_RACE','Pacific Island Male_RACE','Two or  More Male_RACE',
                       'Indian Female_RACE','Asian Female_RACE','Hispanic Female_RACE','Black Female_RACE','White Female_RACE','Pacific Island Female_RACE','Two or  More Female_RACE','Total_RACE'],
              axis=1,inplace=True)

logsvm_df.drop(['lea_not_highqual_class_hp_pct', 'lea_not_highqual_class_lp_pct','lea_not_highqual_class_all_pct'],
              axis=1,inplace=True)

#check it out
len(logsvm_df.columns)

#check null value percentages
NA_report = logsvm_df.apply(lambda x: (sum(x.isnull().values)/len(x))*100, axis = 0)
with pd.option_context('display.max_rows', None):
    print("\n ****** Percentage of missing values in each attributes ********\n\n",NA_report)



#GRADIENT BOOSTING REGRESSION on SPG SCORE	


#preparing dataframes for train-test-split to build models
#separate target variable and predictors
X = logsvm_df.astype(np.float32)
X = X.drop('SPG Score Scaled',axis=1)
y = logsvm_df['SPG Score Scaled']

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#train-test-split, 80% training data
from sklearn.model_selection import train_test_split
indices = range(2419)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y,indices,test_size=0.2, random_state=23233)

#grid search it, bro
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

#gimme those params
n_est = [300,400,500,600]
max_dep = [3,4,5]
min_split = [2,3]
lrn_rate = [0.01,0.05,0.1]
losses = ['ls','lad']

#plug 'em in
parameters = {'n_estimators': n_est, 'max_depth': max_dep, 'min_samples_split': min_split,
          'learning_rate': lrn_rate, 'loss': losses}

#build the classifier, stick in grid search
gbr = ensemble.GradientBoostingRegressor()
clf = GridSearchCV(gbr,parameters,cv=5)
#print(y_train.shape)
#print(X_train.shape)
print(clf)

#fit the model
clf.fit(X_train,y_train)

#one estimator to rule them all
clf.best_estimator_

#generate predictions
predictions = clf.predict(X_test)

#let's see how we did
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

#best parameters
clf.best_params_

#player 1 hi-score
clf.best_score_

#how about all the other scores
clf.grid_scores_

#let's reign in those parameters a bit; probably didn't need all that firepower
params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'ls'}

#create classifier instance with specified parameters
clf_2 = ensemble.GradientBoostingRegressor(**params)

#create indices, in case we need 'em
indices = range(2419)

#train_test lickity-split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=111)

#and fit
clf_2.fit(X_train, y_train)

#how'd we do?
mse = mean_squared_error(y_test, clf_2.predict(X_test))
print("MSE: %.4f" % mse)

# Plot training/testing deviance

#array of zeros in which to put test scores
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

#fill in those test scores at every boosting iteration
for i, y_pred in enumerate(clf_2.staged_predict(X_test)):
    test_score[i] = clf_2.loss_(y_test, y_pred)

#plot it out, with the number of estimators used
plt.figure(figsize=(12, 6))
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf_2.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

#how important were the features? Note: non-parametric regressor means no positive/negative coefficients!
clf_2.feature_importances_

#storing the importances
feature_importance = clf_2.feature_importances_

#scaling importances by their maximum value
feature_importance = 100.0 * (feature_importance / feature_importance.max())

#sorting 'em for neat plotting
sorted_idx = np.argsort(feature_importance)

#gimme a little bit of extra space on the y-axis of that plot
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))

#turn it sideways
plt.barh(pos, feature_importance[sorted_idx], align='center')

#note: gotta make the features an array before we sort 'em
plt.yticks(pos, np.asarray(list(logsvm_df.columns))[sorted_idx])
plt.xlabel('Relative Importance as Percentage of Maximum')
plt.title('Variable Importance')
plt.show()

#let's see that again! -- this time, with raw feature importance values
feature_importance = clf_2.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.asarray(list(logsvm_df.columns))[sorted_idx])
plt.xlabel('Feature Weights')
plt.title('Variable Importance')
plt.show()



#RECURSIVE FEATURE ELIMINATION & REMODELING w/ MOST IMPORTANT FEATURES


#NOTE: GRADIENT BOOSTING REGRESSION IS NON-PARAMETRIC -- NO FEATURE COEFFICIENTS CAN BE RETRIEVED, ONLY FEATURE IMPORTANCES!

#CF: https://stackoverflow.com/questions/47106385/determine-why-features-are-important-in-decision-tree-models

#NOTE ALSO: FURTHER EXPLORATION OF POSITIVE/NEGATIVE FEATURE INFLUENCE ON THE TARGET VARIABLE 
#                  CAN BE CONDUCTED THROUGH PARTIAL DEPEDENCE PLOTS

#CF: http://rstudio-pubs-static.s3.amazonaws.com/283647_c3ab1ccee95a403ebe3d276599a85ab8.html

#(WE WOULD HAVE TO SEE HOW THIS IS DONE IN PYTHON, FOR PURPOSES OF INTERPRETATION)


#let's see if we can pare those features down

#indices and train-test-lickity-split-and-quit
indices = range(2419)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=1111)

#scikit learn is our friend
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

#same parameters as above
params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'ls'}

#build classifier instance w/ specified parameters
clf_3 = ensemble.GradientBoostingRegressor(**params)

#plug it into recursive feature elimination with 10-fold cross-validation and MSE scoring metric
rfecv = RFECV(estimator=clf_3, step=1, cv=KFold(10), scoring='neg_mean_squared_error')

#pipeline is gonna help us retrieve the feature names
#name your classifier and estimator whatever you want, and stick em in tuples
pipeline = Pipeline([
    ('rfe_cv',rfecv),
    ('clf',clf_3)
])

#fit that pipe, bro
pipeline.fit(X_train, y_train)

#how'd we do on the test set?
mse = mean_squared_error(y_test, rfecv.predict(X_test))
print("MSE: %.4f" % mse)

#how many features did we really need?
print("Optimal number of features : %d" % rfecv.n_features_)

#the .named_steps attribute from the pipeline can be indexed with whatever you named your RFECV estimator
#from there you use the .support_ attribute to help you get the feature names
#note: support_feat is just a boolean mask you can apply to the full array of features to get just the ones used by the model
support_feat = pipeline.named_steps['rfe_cv'].support_

#aliasing that full array of features
feat_names = np.array(list(logsvm_df.drop('SPG Score Scaled',axis=1).columns))

#and pulling out the feature names with boolean masking
feat_names[support_feat]

#checking out the curve of those cross-validation scores-- not getting out any extra juice after around 19 features
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#NOW-- let's do it one more time, using just the most important features

#NOTE: (if we were really being statistically vigilant and had tons of data, we'd do this on a hold-out validation set)

#preparing dataframes for train-test-split
#separate target variable and predictors
X = logsvm_df[['tchyrs_0thru3_pct', 'class_teach_num', 'nbpts_num',
       '_1yr_tchr_trnovr_pct', 'highqual_class_pct',
       'lea_tchyrs_11plus_pct', 'lea_class_teach_num',
       'lea_1yr_tchr_trnovr_pct', 'lea_highqual_class_pct',
       'lea_highqual_class_lp_pct', 'lea_highqual_class_all_pct',
       'lea_supplies_expense_pct', 'lea_instruct_equip_exp_pct',
       'White_Pct_RACE', 'Asian_RACE_Pct', 'Hispanic_RACE_Pct',
       'Black_RACE_Pct', 'Black_RACE_Male_Pct', 'White_RACE_Male_Pct']].astype(np.float32)
y = logsvm_df['SPG Score Scaled']

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#same hyperparameters
params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'ls'}

#building that classifier
clf_4 = ensemble.GradientBoostingRegressor(**params)

#same old drill
indices = range(2419)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=333)

#fit, hit, and quit
clf_4.fit(X_train, y_train)

#score it
mse = mean_squared_error(y_test, clf_4.predict(X_test))
print("MSE: %.4f" % mse)

#plot training/testing deviance

#same deal as before
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf_4.staged_predict(X_test)):
    test_score[i] = clf_4.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf_4.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

#plot feature importances scaled to their max value
feature_importance = clf_4.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feat_names[support_feat][sorted_idx])
plt.xlabel('Relative Importance as Percentage of Maximum')
plt.title('Variable Importance')
plt.show()

#plot raw feature importances
feature_importance = clf_4.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feat_names[support_feat][sorted_idx])
plt.xlabel('Feature Weights')
plt.title('Variable Importance')
plt.show()

#This is definitely more interpretable-- but for keener inference, we should still check out partial dependency plots!



#DENSE NEURAL NETWORK CLASSIFICATION of A-GRADE vs. F-GRADE PERFORMANCE



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
#separate target variable and predictors, only selecting records with Grades A = 4 and F = 0
X = logsvm_df[(logsvm_df['SPG Score_GRADE'] == 0) | (logsvm_df['SPG Score_GRADE'] == 4)]
X = X.drop(['SPG Score_GRADE','SPG Score Scaled'],axis=1)
y = logsvm_df['SPG Score_GRADE'][(logsvm_df['SPG Score_GRADE'] == 0) | (logsvm_df['SPG Score_GRADE'] == 4)]

#verifying correct shape
print(X.shape)
print(y.shape)

#checking frequency counts of A-Grade and F-Grade classes
y.value_counts()

#viewing target variable as a dataframe
pd.DataFrame(y).head(10)

#replacing A-Grade value (4) with 1 to facilitate binary classification below
pd.DataFrame(y.replace(to_replace=4, value=1)).head(10)

#performing the above replacement mapping 'in place' to make it permanent
pd.DataFrame(y.replace(to_replace=4, value=1, inplace=True))

#verifying replacement mapping
y[:10]

#too legit 2 (not) train-test-split and quit
indices = range(255)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=555)

#needed imports for DNN classifier script below
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

#Dense Neural Net Binary Classifier (10 X 20 X 10 hidden layers) w/ rectified linear unit activation (500 training steps)

def main():
    
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[50])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=2,
                                          model_dir="/tmp/NN_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_train)},
        y=np.array(y_train),
        num_epochs=None,
        shuffle=True)

    # Train model
    classifier.train(input_fn=train_input_fn, steps=500)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_test)},
        y=np.array(y_test),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    
if __name__ == "__main__":
    main()
