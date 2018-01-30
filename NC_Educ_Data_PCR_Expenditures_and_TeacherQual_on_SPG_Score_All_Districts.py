#Checking to see if PC loadings are similar when PCA applied to all regions
#This should tell us if the relation between the Western and Northeastern Regions identified by PCA is roughly extensible to the data of other regions

#subset the fully merged dataframe by the list of expenditure, teacher qualification, and performance score variables
#(all_df was the alias for the fully merged 'schoolData_testRaceRegion' dataframe from the previous scripts)
fullpc_list_filter = pc_list_nogrades + ['SPG Score Scaled']
fullpc_df = all_df[fullpc_list_filter]
#fullpc_df.shape


#convert all region names to categorical integer values
fullpc_df['SBE District'] = fullpc_df['SBE District'].astype('category')
fullpc_df['SBE District'] = fullpc_df['SBE District'].cat.codes


#scale the SPG Score to a percentage value between 0 and 1, so it can be used as the target value in PC regression (PCR)
#fullpc_df['SPG Score Scaled'] = fullpc_df['SPG Score']*(.01)
#pd.Series.to_frame(fullpc_df['SPG Score Scaled']).shape

#by region, groupwise mean fill of missing scaled SPG Scores 
#mean_fill = fullpc_df.groupby('SBE District')['SPG Score Scaled'].transform('mean')
#fullpc_df['SPG Score Scaled'] = fullpc_df['SPG Score Scaled'].fillna(mean_fill)


#overall mean impute the missing values for the remaining schools which did not report a region by which to mean impute SPG Score; necessary for PCR
#fullpc_df['SPG Score Scaled'] = pd.Series.to_frame(fullpc_df['SPG Score Scaled']).fillna(pd.Series.to_frame(fullpc_df['SPG Score Scaled']).mean())


#perform PCA on the new full dataframe
#mean impute any other missing values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(fullpc_df)
imp_data = imp.transform(fullpc_df)

#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(imp_data)
scaled_data = scaler.transform(imp_data)

#perform PCA with 16 components (it will be shown below that only the first two PCs really matter, as was the case when comparing Western and Northeastern Regions)
#A clear elbow is identifiable after 2 or 3 PCs
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape



#prepare the data for PC Regression
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression

all_df['SPG Score Scaled'] = all_df['SPG Score']*(.01)
#pd.Series.to_frame(all_df['SPG Score Scaled']).shape

#region-wise mean imputation of SPG Scores
mean_fill = all_df.groupby('SBE District')['SPG Score Scaled'].transform('mean')
all_df['SPG Score Scaled'] = all_df['SPG Score Scaled'].fillna(mean_fill)

#mean impute all remaining SPG Scores (for records w/o regions, mean impute across all regions)
all_df['SPG Score Scaled'] = pd.Series.to_frame(all_df['SPG Score Scaled']).fillna(pd.Series.to_frame(all_df['SPG Score Scaled']).mean())

#set the scaled SPG Score to the target variable for PCR
y = all_df['SPG Score Scaled']

#create instance of KFold cross-validation class, using the dimension of the PCA transformed data, randomly shuffling, and setting k = 10
n = len(x_pca)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True,random_state=101)

#create instance of linear regression model class, and empty list to receive MSE values
regr = LinearRegression()
mse = []

#generating cross validation MSE scores for the model intercept (with dimensions: (PCA transformed data) x 1), flattening the target values to 1D array of the same length as 1st dimension of PCA transformed data
score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score)

#performing PCR, adding one component to the regression at a time
for i in np.arange(1,17):
    score = -1*cross_validation.cross_val_score(regr,x_pca[:,:i], y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)

#creating two subplots, one for the results of PCR without the intercept, and one which includes the intercept
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(mse, '-v') #includes the intercept term
ax2.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], mse[1:17], '-v') #plotting MSE of PCs one at a time, excluding the intercept
ax2.set_title('Intercept excluded from plot')

#labeling axes and setting axis limits
for ax in fig.axes:
    ax.set_xlabel('Number of principal components in regression')
    ax.set_ylabel('MSE')
    ax.set_xlim((-0.2,16.2))
	


#Comparing with results of Partial Least Squares Regression: cf. https://www.mathworks.com/help/stats/examples/partial-least-squares-regression-and-principal-components-regression.html?requestedDomain=true
#PLSR and PCR are both methods to model a response variable when there are a large number of predictor variables, 
#	and those predictors are highly correlated or even collinear. 
#Both methods construct new predictor variables, known as components, as linear combinations of the original predictor variables, 
#	but they construct those components in different ways. 
#PCR creates components to explain the observed variability in the predictor variables, without considering the response variable at all. 
#	On the other hand, PLSR does take the response variable into account, and therefore often leads to models that are able to fit 
#	the response variable with fewer components.

#Partial Least Squares Regression
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression

#empty list for MSE values
mse = []

#creating K-fold cross-validation class instance, randomly shuffling, and setting k = 10
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=101)

#performing partial least squares regression for all 16 PCs
for i in np.arange(1, 17):
    pls = PLSRegression(n_components=i, scale=False) #create PLSR class instance
    pls.fit(scale(x_pca),y) #ensure x data is scaled before fitting model
    score = cross_validation.cross_val_score(pls, x_pca, y, cv=kf_10, scoring='neg_mean_squared_error').mean() #generating MSE scores
    mse.append(-score) #reducing MSE after every PC

plt.plot(np.arange(1, 17), np.array(mse), '-v') #plotting MSE for PCs one at a time
plt.xlabel('Number of principal components in PLS regression') #setting axis labels
plt.ylabel('MSE')
plt.xlim((-0.2, 16.2)) #setting axis limits







#adding the scaled SPG score to the full list of variables used in PCA
fullpc_list = pc_list_nogrades + ['SPG Score Scaled']

#flattening the full list of PCA variables (used in creating the heatmap)
flatten_fullpc_list = pd.DataFrame({'vars':fullpc_list})
flat_fullpc_list = flatten_fullpc_list.values.flatten()
flat_fullpc_list.shape

#return dataframe with just the first 2 PCs (from PCA, PCR, and PLSR above, it's clear that only the first 2 PCs really matter, even when comparing across all regions)
fulldf_comp = pd.DataFrame(pca.components_[:2],columns=flat_fullpc_list)
#with pd.option_context('display.max_columns', None):
#    print(df_comp)

#the pattern of the heatmap for the first 2 PCs closely resembles that which was found in performing PCA on just the data from the Western and Northeastern Regions
plt.figure(figsize=(20,8))
sns.heatmap(fulldf_comp,cmap='plasma')






#check to make sure the conversion of region values to categorical integers was successful
#fullpc_df['SBE District'].value_counts()

#compare with the value counts of the region data before categorical conversion, to see which values correspond to which regions
#all_df['SBE District'].value_counts()

#create scatterplot with all of the regions on the same axes of the first 2 PC loadings
#this will look slightly different from the scatterplot created when doing PCA on just the Western and Northeastern Regions, but the overall pattern is quite similar
#note that the Northeastern Region is pretty anomalous on the axis of that first PC
pc_1 = x_pca[:,0]
pc_2 = x_pca[:,1]
fullpc_region = fullpc_df['SBE District']
fulltest_pc_df = pd.DataFrame(dict(c0=pc_1,c1=pc_2,c2=fullpc_region)) #creating a dataframe whose mappin can be iterated over when generating the scatterplot, grouped by region

#color mapping the different regions to the corresponding categorical integer codes
colors = {1:'yellow',7:'blue',2:'green',3:'red',4:'purple',5:'orange',6:'cyan',0:'magenta',-1:'black'}

#plotting all the regions on the same 2 PC axes
fig,ax = plt.subplots()
for key,group in fulltest_pc_df.groupby('c2'):
    group.plot.scatter(ax=ax,x='c0',y='c1',label=key,color=colors[key],alpha=0.5)

#labeling and creating the plot legend
plt.xlabel('First PC')
plt.ylabel('Second PC')
L = plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5)) #this puts the legend outside the plot
L.get_texts()[0].set_text('Unknown Region')
L.get_texts()[1].set_text('North Central Region')
L.get_texts()[2].set_text('Northeast Region')
L.get_texts()[3].set_text('Northwest Region')
L.get_texts()[4].set_text('Piedmont Triad Region')
L.get_texts()[5].set_text('SandHills Region')
L.get_texts()[6].set_text('Southeast Region')
L.get_texts()[7].set_text('Southwest Region')
L.get_texts()[8].set_text('Western Region')

