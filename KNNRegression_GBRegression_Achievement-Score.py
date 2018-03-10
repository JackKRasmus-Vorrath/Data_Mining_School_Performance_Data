#KNN REGRESSION on OVERALL ACHIEVEMENT SCORE


#feature list from cleaning procedure above
len(logsvm_list_noDistrict_race)

#adding target variable to feature list
feature_mask = logsvm_list_noDistrict_race + ['Overall Achievement Score']
len(feature_mask)

#subsetting cleaned data with feature list masking
logsvm_df2 = all_df_nocharter[feature_mask]
logsvm_df2.shape

#check null value percentages
NA_report = logsvm_df2.apply(lambda x: (sum(x.isnull().values)/len(x))*100, axis = 0)
with pd.option_context('display.max_rows', None):
    print("\n ****** Percentage of missing values in each attributes ********\n\n",NA_report)
	
#drop rows with null values for target variable
knnR_df = logsvm_df2[logsvm_df2['Overall Achievement Score'].notnull()]

#check shape of final dataframe
knnR_df.shape

#scale the Overall Achievement Score feature
knnR_df['Overall Achievement Score Scaled'] = knnR_df['Overall Achievement Score']*(.01)
knnR_df.shape

#checking result of data type conversion below
type(knnR_df['Overall Achievement Score Scaled'].iloc[0].astype(np.float32))

#training feature data type conversion; dropping targets from training dataframe; setting target variable
knnR_df2 = knnR_df.astype(np.float32)
X = knnR_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled'],axis=1)
y = knnR_df['Overall Achievement Score Scaled'].astype(np.float32)

#verify correct shape
print(X.shape)
print(y.shape)

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#train-test-split, 80% training data
from sklearn.model_selection import train_test_split
indices = range(2185)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=2222)

#classifier, grid search, and metric imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#gimme those params
n_neigh = [1,3,5]
weightz = ['uniform','distance']
algo = ['ball_tree','kd_tree']
leaf_sz = [20,30,50]
metrik = ['minkowski','euclidean']

#plug 'em in
parameters = {'n_neighbors': n_neigh, 'weights': weightz, 'algorithm': algo,
          'leaf_size': leaf_sz, 'metric': metrik}
		  
#build the classifier, stick in grid search
knnR = KNeighborsRegressor()
clf = GridSearchCV(knnR,parameters,cv=5)
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

#refining the hyperparameters
params = {'n_neighbors': 7, 'weights': 'distance', 'algorithm': 'ball_tree',
          'leaf_size': 20, 'metric': 'minkowski'}

#create classifier instance with specified parameters
clf_3 = KNeighborsRegressor(**params)

#create indices, in case we need 'em
indices = range(2185)

#train_test lickity-split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=333)

#and fit
clf_3.fit(X_train, y_train)

#how'd we do?
mse = mean_squared_error(y_test, clf_3.predict(X_test))
print("MSE: %.4f" % mse)

#R-squared coefficient of determination
clf_3.score(X_test, y_test, sample_weight=None)

#checking out the KNN graph
#np.set_printoptions(threshold=np.nan)

#KNN_graph = clf_3.kneighbors_graph(X_test)
#KNN_graph.toarray()




##############################################################

#GRADIENT BOOSTING REGRESSION on OVERALL ACHIEVEMENT SCORE


#drop rows with null values for target variable
gbr_df = logsvm_df2[logsvm_df2['Overall Achievement Score'].notnull()]

#check shape of final dataframe
gbr_df.shape

#scale the Overall Achievement Score feature
gbr_df['Overall Achievement Score Scaled'] = gbr_df['Overall Achievement Score']*(.01)
gbr_df.shape

#checking result of data type conversion below
type(gbr_df['Overall Achievement Score Scaled'].iloc[0].astype(np.float32))

#training feature data type conversion; dropping targets from training dataframe; setting target variable
gbr_df2 = gbr_df.astype(np.float32)
X = gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled'],axis=1)
y = gbr_df['Overall Achievement Score Scaled'].astype(np.float32)

#verify correct shape
print(X.shape)
print(y.shape)

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#train-test-split, 80% training data
from sklearn.model_selection import train_test_split
indices = range(2185)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=1234)

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
clf_5 = GridSearchCV(gbr,parameters,cv=5)
#print(y_train.shape)
#print(X_train.shape)
print(clf_5)

#fit the model
clf_5.fit(X_train,y_train)

#one estimator to rule them all
clf_5.best_estimator_

#generate predictions
predictions = clf_5.predict(X_test)

#let's see how we did
mse = mean_squared_error(y_test, clf_5.predict(X_test))
print("MSE: %.4f" % mse)

#best parameters
clf_5.best_params_

#player 1 hi-score
clf_5.best_score_

#refining hyperparameters
params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'ls'}

#create classifier instance with specified parameters
clf_6 = ensemble.GradientBoostingRegressor(**params)

#create indices, in case we need 'em
indices = range(2185)

#train_test lickity-split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=4444)

#and fit
clf_6.fit(X_train, y_train)

#WARNING: We're doing suspiciously well...! (See below)
mse = mean_squared_error(y_test, clf_6.predict(X_test))
print("MSE: %.4f" % mse)

# Plot training/testing deviance

#array of zeros in which to put test scores
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

#fill in those test scores at every boosting iteration
for i, y_pred in enumerate(clf_6.staged_predict(X_test)):
    test_score[i] = clf_6.loss_(y_test, y_pred)

#plot it out, with the number of estimators used
plt.figure(figsize=(12, 6))
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf_6.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

#how important were the features? Note: non-parametric regressor means no positive/negative coefficients!
clf_6.feature_importances_

#storing the importances
feature_importance = clf_6.feature_importances_

#scaling importances by their maximum value
feature_importance = 100.0 * (feature_importance / feature_importance.max())

#sorting 'em for neat plotting
sorted_idx = np.argsort(feature_importance)

#gimme a little bit of extra space on the y-axis of that plot
pos = np.arange(sorted_idx.shape[0]) + .5

#NOTE: the SPG Score totally overpowers the regressor, and is not likely to be available as a feature in a real-life use-case

#Meaning, the SPG Score is a good proxy for the Overall Achievement Score

#However, if we had the one score, we probably wouldn't need to figure out the other to get a decent gauge on performance!

plt.figure(figsize=(12, 12))

#turn it sideways
plt.barh(pos, feature_importance[sorted_idx], align='center')

#note: gotta make the features an array before we sort 'em
plt.yticks(pos, np.asarray(list(gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled'],axis=1).columns))[sorted_idx])
plt.xlabel('Relative Importance as Percentage of Maximum')
plt.title('Variable Importance')
plt.show()

#let's see that again! -- this time, with raw feature importance values
feature_importance = clf_6.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.asarray(list(gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled'],axis=1).columns))[sorted_idx])
plt.xlabel('Feature Weights')
plt.title('Variable Importance')
plt.show()

#This time, let's get rid of the overpowering SPG Score and use recursive feature elimination

#training feature data type conversion; dropping targets from training dataframe; setting target variable
gbr_df2 = gbr_df.astype(np.float32)
X = gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled','SPG Score Scaled'],axis=1)
y = gbr_df['Overall Achievement Score Scaled'].astype(np.float32)

#verify correct shape
print(X.shape)
print(y.shape)

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#train-test-split, 80% training data
from sklearn.model_selection import train_test_split
indices = range(2185)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=2345)

#scikit learn is our friend
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

#same optimal hyperparameters identified by GridSearchCV
params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'ls'}

#create classifier instance with specified parameters
clf_7 = ensemble.GradientBoostingRegressor(**params)

#plug it into recursive feature elimination with 10-fold cross-validation and MSE scoring metric
rfecv = RFECV(estimator=clf_7, step=1, cv=KFold(10), scoring='neg_mean_squared_error')

#pipeline is gonna help us retrieve the feature names
#name your classifier and estimator whatever you want, and stick em in tuples
pipeline = Pipeline([
    ('rfe_cv',rfecv),
    ('clf',clf_7)
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
feat_names = np.array(list(gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled','SPG Score Scaled'],axis=1).columns))

#and pulling out the feature names with boolean masking
feat_names[support_feat]

#checking out the curve of those cross-validation scores

#NOTE: arguably, we could have chosen around 20 features and done equally well

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#let's pull out the raw feature importance values to just get the best 20 or so predictors

#NOTE: since the school district can wield no influence over which demographics attend which schools, let's pull out those
#      most important features toward which they might usefully direct their efforts and see how the model performs

feature_importance = clf_7.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.asarray(list(gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled','SPG Score Scaled'],axis=1).columns))[sorted_idx])
plt.xlabel('Feature Weights')
plt.title('Variable Importance')
plt.show()

#NOW-- let's do it one more time, using just the most important and actionable features

#preparing dataframes for train-test-split
#separate target variable and predictors
X = gbr_df2[['class_teach_num','lea_class_teach_num','tchyrs_11plus_pct','advance_dgr_pct','lea_not_highqual_class_all_pct',
             'lea_1yr_tchr_trnovr_pct','lea_tchyrs_0thru3_pct','lea_advance_dgr_pct','tchyrs_4thru10_pct',
             '_1yr_tchr_trnovr_pct','lea_tchyrs_11plus_pct','lea_tchyrs_4thru10_pct','lea_flicensed_teach_pct',
             'lea_highqual_class_pct','tchyrs_0thru3_pct','flicensed_teach_pct','lea_highqual_class_lp_pct',
             'lea_total_expense_num','highqual_class_pct']].astype(np.float32)
y = gbr_df['Overall Achievement Score Scaled']

print(X.shape)
print(y.shape)

#scale training predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#same optimal hyperparameters identified by GridSearchCV
params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'ls'}

#building that classifier
clf_8 = ensemble.GradientBoostingRegressor(**params)

#same old drill
indices = range(2185)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=456)

#fit, hit, and quit
clf_8.fit(X_train, y_train)

#NOTE: doesn't perform as well, but the model is probably more interpretable and useful
mse = mean_squared_error(y_test, clf_8.predict(X_test))
print("MSE: %.4f" % mse)

#plot training/testing deviance

#same deal as before
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf_8.staged_predict(X_test)):
    test_score[i] = clf_8.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf_8.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

#plot feature importances scaled to their max value
feature_importance = clf_8.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.asarray(list(gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled','SPG Score Scaled'],axis=1).columns))[sorted_idx])
plt.xlabel('Relative Importance as Percentage of Maximum')
plt.title('Variable Importance')
plt.show()

#plot raw feature importances
feature_importance = clf_8.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.asarray(list(gbr_df2.drop(['Overall Achievement Score','Overall Achievement Score Scaled','SPG Score Scaled'],axis=1).columns))[sorted_idx])
plt.xlabel('Feature Weights')
plt.title('Variable Importance')
plt.show()


#As before, retaining well credentialed, fully licensed, and experienced teachers to deliver high quality classes is key!	  
