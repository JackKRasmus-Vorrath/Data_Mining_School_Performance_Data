import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#Read in raw data
schoolData = pd.read_csv('C:/Users/jkras/Desktop/All_Data_By_School_Final.csv', low_memory=False)
testScores = pd.read_csv('C:/Users/jkras/Desktop/1516testresults_masking_removed.csv', low_memory=False)
raceData = pd.read_csv('C:/Users/jkras/Desktop/Ec_Pupils_Expanded (2017 Race Compositions by School).csv', low_memory=False)


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
logsvm_list_excludeDistrict = teach_list + exp_list + ['Overall Achievement Score'] + ['SPG Score Scaled'] + ['Northeast Region', 'Northwest Region', 'Piedmont Triad Region', 'SandHills Region', 'Southeast Region', 'Southwest Region', 'Western Region']


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

#dropping target variable to prepare feature dataframe for final two modeling tasks at the end of the notebook
logsvm_df_no_OAS = logsvm_df.drop('Overall Achievement Score',axis=1)
len(logsvm_df_no_OAS)

logsvm_df3 = logsvm_df_no_OAS.drop('SPG Score Scaled',axis=1)

cluster_df = pd.concat([logsvm_df3, logsvm_df_no_OAS['SPG Score Scaled']], axis=1)

#create categorization bins and names for scaled SPG Score
bins = [0.0,0.55,1.0]
group_names = ['F/D','Other']

#create SPG Score Grade feature, based on binned SPG Score
cluster_df['SPG Score_GRADE'] = pd.cut(cluster_df['SPG Score Scaled'],bins,labels=group_names,include_lowest=True,right=False,precision=6)

#generate categorical codes: 0 = F,D ---- 1 = C,B,A
cluster_df['SPG Score_GRADE'] = cluster_df['SPG Score_GRADE'].cat.codes

cluster_df_clean = cluster_df.drop('SPG Score Scaled',axis=1)


#features shown to have the heaviest weights in the classification (identified through first pass PCA, not shown)
cluster_df_refined_2 = cluster_df_clean[['flicensed_teach_pct', 'tchyrs_0thru3_pct', 'tchyrs_11plus_pct',
       'class_teach_num', 'nbpts_num', 'advance_dgr_pct',
       '_1yr_tchr_trnovr_pct', 'highqual_class_pct', 'lea_flicensed_teach_pct',
       'lea_tchyrs_0thru3_pct', 'lea_tchyrs_11plus_pct', 'lea_class_teach_num',
       'lea_nbpts_num', 'lea_advance_dgr_pct', 'lea_1yr_tchr_trnovr_pct',
       'lea_lateral_teach_pct', 'lea_highqual_class_pct',
       'lea_highqual_class_all_pct', 'lea_total_expense_num',
       'lea_salary_expense_pct', 'lea_benefits_expense_pct',
       'lea_supplies_expense_pct', 'lea_instruct_equip_exp_pct',
       'White_Pct_RACE', 'Majority_Minority_RACE', 'Asian_RACE_Pct',
       'Hispanic_RACE_Pct', 'Black_RACE_Pct','SPG Score_GRADE']]
	
#resetting the index to avoid subsetting errors, moving forward
cluster_df_refined_2_fixed_index = cluster_df_refined_2.reset_index(drop=True)
#cluster_df_refined_2_fixed_index.index



#####KNNClassification -- SPG SCORE D/F vs. A/B/C#########	

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#fit scaler to features
scaler.fit(cluster_df_refined_2_fixed_index.drop('SPG Score_GRADE',axis=1))

#transform features to scaled version
scaled_features = scaler.transform(cluster_df_refined_2_fixed_index.drop('SPG Score_GRADE',axis=1))

#convert scaled features to df
df_feat = pd.DataFrame(scaled_features,columns=cluster_df_refined_2_fixed_index.columns[:-1])
#df_feat.head(3)

#train/test split
from sklearn.cross_validation import train_test_split

X = df_feat
y = cluster_df_refined_2_fixed_index['SPG Score_GRADE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1242)



#import classifier
from sklearn.neighbors import KNeighborsClassifier

#elbow method for choosing best K value
error_rate = []

for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance',algorithm='ball_tree') #iterate through k-values
    knn.fit(X_train,y_train) #model fitting for every k-value
    pred_i = knn.predict(X_test) #test set predictions for given k-value
    error_rate.append(np.mean(pred_i != y_test)) #append error rates for each k to list

#plot K-value vs. error rates
plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K-Value')
plt.xlabel('K')
plt.ylabel('Error Rate')



#create model instance, k = 31
knn = KNeighborsClassifier(n_neighbors=31, weights='distance', algorithm='ball_tree',metric='minkowski')

#fit KNN model to training data
knn.fit(X_train,y_train)

#predict method to generate predictions from KNN model and test data
pred = knn.predict(X_test)

#create confusion matrix and classification report
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))



########Model Evaluation##########

from sklearn import metrics

###Place the below commentary in a markdown cell###

#If the ground truth labels are not known, evaluation must be performed using the model itself. 
#The Silhouette Coefficient (sklearn.metrics.silhouette_score) is an example of such an evaluation, 
#where a higher Silhouette Coefficient score relates to a model with better defined clusters. 
#The Silhouette Coefficient is defined for each sample and is composed of two scores:

#a: The mean distance between a sample and all other points in the same class.
#b: The mean distance between a sample and all other points in the next nearest cluster.

#The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering
#Scores around zero indicate overlapping clusters
#The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

metrics.silhouette_score(X, y)


#Given the knowledge of the ground truth class assignments labels_true and our clustering algorithm assignments 
#of the same samples labels_pred, the Mutual Information is a function that measures the agreement of the two assignments, 
#ignoring permutations. Two different normalized versions of this measure are available, 
#Normalized Mutual Information(NMI) and Adjusted Mutual Information(AMI). 
#NMI is often used in the literature while AMI was proposed more recently and is normalized against chance

#Perfect labeling is scored 1.0
#Bad (e.g. independent labelings) have non-positive scores

#Random (uniform) label assignments have a AMI score close to 0.0 for any value of n_clusters and n_samples 
#(which is not the case for raw Mutual Information or the V-measure for instance).
#Bounded range [0, 1]: Values close to zero indicate two label assignments that are largely independent, 
#while values close to one indicate significant agreement. 
#Further, values of exactly 0 indicate purely independent label assignments 
#and a AMI of exactly 1 indicates that the two label assignments are equal (with or without permutation).

metrics.adjusted_mutual_info_score(y_test,pred)


#Given the knowledge of the ground truth class assignments labels_true 
#and our clustering algorithm assignments of the same samples labels_pred, 
#the adjusted Rand index is a function that measures the similarity of the two assignments, 
#ignoring permutations and with chance normalization

#Perfect labeling is scored 1.0
#Bad (e.g. independent labelings) have negative or close to 0.0 scores

#Random (uniform) label assignments have a ARI score close to 0.0 for any value of n_clusters and n_samples 
#(which is not the case for raw Rand index or the V-measure for instance)
#Bounded range [-1, 1]: negative values are bad (independent labelings), 
#similar clusterings have a positive ARI, 1.0 is the perfect match score

metrics.adjusted_rand_score(y_test,pred)

#Given the knowledge of the ground truth class assignments of the samples, 
#it is possible to define some intuitive metric using conditional entropy analysis.

#homogeneity: each cluster contains only members of a single class.
#completeness: all members of a given class are assigned to the same cluster.

#We can turn those concept as scores homogeneity_score and completeness_score. 
#Both are bounded below by 0.0 and above by 1.0 (higher is better)

#Their harmonic mean called V-measure is computed by v_measure_score
#The V-measure is actually equivalent to the mutual information (NMI) discussed above normalized by the sum of the label entropies [B2011].

#Homogeneity, completeness and V-measure can be computed at once using 'homogeneity_completeness_v_measure'

#Bounded scores: 0.0 is as bad as it can be, 1.0 is a perfect score.
#Intuitive interpretation: clustering with bad V-measure can be qualitatively analyzed in terms of homogeneity and completeness to better feel what ‘kind’ of mistakes is done by the assignment.

print(metrics.homogeneity_score(y_test,pred))
print(metrics.completeness_score(y_test,pred))
print(metrics.v_measure_score(y_test,pred))

metrics.homogeneity_completeness_v_measure(y_test,pred)

#The previously introduced metrics are not normalized with regards to random labeling: 
#this means that depending on the number of samples, clusters and ground truth classes, 
#a completely random labeling will not always yield the same values for homogeneity, 
#completeness and hence v-measure. 
#In particular random labeling won’t yield zero scores especially when the number of clusters is large.

#This problem can safely be ignored when the number of samples is more than a thousand 
#and the number of clusters is less than 10. 
#For smaller sample sizes or larger number of clusters 
#it is safer to use an adjusted index such as the Adjusted Rand Index (ARI)



#########KNN Classifier Model Visualization###########

#features column list created for PCA visualization
feat_cols = ['flicensed_teach_pct', 'tchyrs_0thru3_pct', 'tchyrs_11plus_pct',
       'class_teach_num', 'nbpts_num', 'advance_dgr_pct',
       '_1yr_tchr_trnovr_pct', 'highqual_class_pct', 'lea_flicensed_teach_pct',
       'lea_tchyrs_0thru3_pct', 'lea_tchyrs_11plus_pct', 'lea_class_teach_num',
       'lea_nbpts_num', 'lea_advance_dgr_pct', 'lea_1yr_tchr_trnovr_pct',
       'lea_lateral_teach_pct', 'lea_highqual_class_pct',
       'lea_highqual_class_all_pct', 'lea_total_expense_num',
       'lea_salary_expense_pct', 'lea_benefits_expense_pct',
       'lea_supplies_expense_pct', 'lea_instruct_equip_exp_pct',
       'White_Pct_RACE', 'Majority_Minority_RACE', 'Asian_RACE_Pct',
       'Hispanic_RACE_Pct', 'Black_RACE_Pct']

#flattening the full list of PCA variables (used in creating the heatmap)
flatten_pc_list = pd.DataFrame({'vars':feat_cols})
flat_pc_list = flatten_pc_list.values.flatten()
flat_pc_list.shape

#perform PCA with 2 Components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)

df_comp = pd.DataFrame(pca.components_[:2],columns=flat_pc_list)

plt.figure(figsize=(20,8))
sns.heatmap(df_comp,cmap='coolwarm')



#######2D Visualization using PCA############

y = cluster_df_refined_2_fixed_index['SPG Score_GRADE']

#step size of the mesh grid
h = .02

#optimal number of neighbors
n_neighbors = 31

#color maps
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

from sklearn import neighbors

for weights in ['uniform', 'distance']:
    #create instance of KNNClassifier and fit
    clf = neighbors.KNeighborsClassifier(n_neighbors, algorithm='ball_tree', weights=weights)
    clf.fit(x_pca, y)

    #plot decision boundary; assign color to each
    #point in the mesh [x_min, x_max] x [y_min, y_max]
    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    #put result in color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    #plot training points also
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


#######3D Visualization using PCA############


#perform PCA with 3 Components
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
x_pca = pca.transform(X)

df_comp = pd.DataFrame(pca.components_[:3],columns=flat_pc_list)

plt.figure(figsize=(20,8))
sns.heatmap(df_comp,cmap='coolwarm')


###3D Interactive Visualization using 3 Principal Components and KMeans++ Clustering###

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np

from sklearn.cluster import KMeans
from sklearn import datasets

print(np.amax(x_pca, axis=0))

print(np.amin(x_pca, axis=0))


np.random.seed(5)

fig = tools.make_subplots(rows=2, cols=1,
                          print_grid=False,
                          specs=[[{'is_3d': True}],
                                [{'is_3d': True}]])
scene = dict(
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2.5, y=0.1, z=0.1)
    ),
    xaxis=dict(
        range=[-6, 17],
        title='PC_1',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    ),
    yaxis=dict(
        range=[-7, 10],
        title='PC_2',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    ),
    zaxis=dict(
        range=[-4,11],
        title='PC_3',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    )
)

centers = [[1, 1], [-1, -1], [1, -1]]
X = x_pca
y = cluster_df_refined_2['SPG Score_GRADE']

estimators = {'k_means_2': KMeans(n_clusters=2, init='k-means++',random_state=1)
              }
fignum = 1
for name, est in estimators.items():
    est.fit(X)
    labels = est.labels_

    trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                         showlegend=False,
                         mode='markers',
                         marker=dict(
                                color=labels.astype(np.float),
                                line=dict(color='black', width=1)
        ))
    fig.append_trace(trace, 1, fignum)
    
    fignum = fignum + 1

y = np.choose(y, [0,1,2]).astype(np.float)

trace1 = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                      showlegend=False,
                      mode='markers',
                      marker=dict(
                            color=y,
                            line=dict(color='black', width=1)))
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=1400, width=1000,
                     margin=dict(l=10,r=10))

fig['layout']['scene1'].update(scene)
fig['layout']['scene2'].update(scene)
fig['layout']['scene3'].update(scene)
fig['layout']['scene4'].update(scene)
fig['layout']['scene5'].update(scene)

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import cufflinks as cf

#connects JS to notebook so plots work inline
init_notebook_mode(connected=True)

#allow offline use of cufflinks
cf.go_offline()

iplot(fig)

metrics.silhouette_score(X, y)



###3D Interactive Visualization using 3 Principal Components and DBSCAN Clustering -- Northeastern (Poor Performing) vs. Western (Well Performing) Region###

clust_df = pd.concat([logsvm_df3, logsvm_df_no_OAS['SPG Score Scaled'], logsvm_df['Overall Achievement Score']], axis=1)
#clust_df.head()
#clust_df.shape

#resetting the index to avoid subsetting errors, moving forward
clust_df = clust_df.reset_index(drop=True)
#cluster_df_refined_2_fixed_index.index

#scale the SPG Score feature
clust_df['Overall Achievement Score Scaled'] = clust_df['Overall Achievement Score']*(.01)

clust_df.drop(['Overall Achievement Score'],axis=1,inplace=True)

clust_df_region = clust_df[['flicensed_teach_pct', 'tchyrs_0thru3_pct', 'tchyrs_4thru10_pct',
       'tchyrs_11plus_pct', 'class_teach_num', 'nbpts_num', 'advance_dgr_pct',
       '_1yr_tchr_trnovr_pct', 'highqual_class_pct', 'lea_flicensed_teach_pct',
       'lea_tchyrs_0thru3_pct', 'lea_tchyrs_4thru10_pct',
       'lea_tchyrs_11plus_pct', 'lea_class_teach_num', 'lea_nbpts_num',
       'lea_advance_dgr_pct', 'lea_1yr_tchr_trnovr_pct',
       'lea_lateral_teach_pct', 'lea_highqual_class_pct',
       'lea_highqual_class_hp_pct', 'lea_highqual_class_lp_pct',
       'lea_highqual_class_all_pct', 'lea_total_expense_num',
       'lea_salary_expense_pct', 'lea_benefits_expense_pct',
       'lea_services_expense_pct', 'lea_supplies_expense_pct',
       'lea_instruct_equip_exp_pct', 'Northeast Region','Western Region']]
	   
#0 = Northeast Region, 1 = Western Region
clust_df_region['Region'] = np.where(clust_df_region['Northeast Region'] == 1, 0, 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#fit scaler to features
scaler.fit(clust_df_region.drop(['Northeast Region', 'Western Region','Region'],axis=1))

#transform features to scaled version
scaled_features = scaler.transform(clust_df_region.drop(['Northeast Region', 'Western Region','Region'],axis=1))

#convert scaled features to df
df_feat = pd.DataFrame(scaled_features,columns=clust_df_region.columns[:-3])

X = df_feat

#perform PCA with 3 Components
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
x_pca = pca.transform(X)

print(np.amax(x_pca, axis=0))
print(np.amin(x_pca, axis=0))


from sklearn.cluster import DBSCAN

np.random.seed(5)

fig = tools.make_subplots(rows=2, cols=1,
                          print_grid=False,
                          specs=[[{'is_3d': True}],
                                [{'is_3d': True}]])
scene = dict(
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2.5, y=0.1, z=0.1)
    ),
    xaxis=dict(
        range=[-6, 17],
        title='PC_1',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    ),
    yaxis=dict(
        range=[-8, 8],
        title='PC_2',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    ),
    zaxis=dict(
        range=[-5,10],
        title='PC_3',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    )
)

centers = [[1, 1], [-1, -1], [1, -1]]
X = x_pca
y = clust_df_region['Region']

estimators = {'dbscan': DBSCAN(eps=1.9, min_samples=15).fit(X)
              }
fignum = 1
for name, est in estimators.items():
    est.fit(X)
    labels = est.labels_

    trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                         showlegend=False,
                         mode='markers',
                         marker=dict(
                                color=labels.astype(np.float),
                                line=dict(color='black', width=1)
        ))
    fig.append_trace(trace, 1, fignum)
    
    fignum = fignum + 1

y = np.choose(y, [0,1]).astype(np.float)

trace1 = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                      showlegend=False,
                      mode='markers',
                      marker=dict(
                            color=y,
                            line=dict(color='black', width=1)))
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=1400, width=1000,
                     margin=dict(l=10,r=10))

fig['layout']['scene1'].update(scene)
fig['layout']['scene2'].update(scene)
fig['layout']['scene3'].update(scene)
fig['layout']['scene4'].update(scene)
fig['layout']['scene5'].update(scene)


iplot(fig)

metrics.silhouette_score(X, y)




######AFFINITY PROPAGATION CLUSTERING -- SPG SCORE D/F vs. A/B/C#########	

#removing one feature with significantly different scale
aff_prop_df = cluster_df_refined_2_fixed_index.drop(['lea_total_expense_num'],axis=1)
aff_prop_df.shape	 


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#fit scaler to features
scaler.fit(aff_prop_df.drop('SPG Score_GRADE',axis=1))

#transform features to scaled version
scaled_features = scaler.transform(aff_prop_df.drop('SPG Score_GRADE',axis=1))

#convert scaled features to df
df_feat = pd.DataFrame(scaled_features,columns=aff_prop_df.columns[:-1])


aff_prop_df_final = pd.concat([df_feat, aff_prop_df['SPG Score_GRADE']],axis=1)
aff_prop_df_final.shape



clus_aff_ABC = aff_prop_df_final[aff_prop_df_final['SPG Score_GRADE'] == 1]
clus_aff_DF = aff_prop_df_final[aff_prop_df_final['SPG Score_GRADE'] == 0]
#clus_aff_DF.head()

clus_aff_ABC_feat = clus_aff_ABC.drop(['SPG Score_GRADE'],axis=1)
clus_aff_DF_feat = clus_aff_DF.drop(['SPG Score_GRADE'],axis=1)
print(clus_aff_ABC_feat.shape)
print(clus_aff_DF_feat.shape)

X_ABC = clus_aff_ABC_feat
X_DF = clus_aff_DF_feat

#perform PCA with 1 Component
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X_ABC)
x_pca_ABC = pca.transform(X_ABC)

#perform PCA with 1 Component
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X_DF)
x_pca_DF = pca.transform(X_DF)

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_completeness_v_measure

#######PLACE THE BELOW DESCRIPTION IN A MARKDOWN CELL TO SEE THE FORMULAS#######

## From http://scikit-learn.org/stable/modules/clustering.html#affinity-propagation

### AffinityPropagation creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at which point the final exemplars are chosen, and hence the final clustering is given.

### Affinity Propagation can be interesting as it chooses the number of clusters based on the data provided. For this purpose, the two important parameters are the preference, which controls how many exemplars are used, and the damping factor which damps the responsibility and availability messages to avoid numerical oscillations when updating these messages.

### The main drawback of Affinity Propagation is its complexity. The algorithm has a time complexity of the order O(N^2 T), where N is the number of samples and T is the number of iterations until convergence. Further, the memory complexity is of the order O(N^2) if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used. This makes Affinity Propagation most appropriate for small to medium sized datasets.

### Algorithm description: The messages sent between points belong to one of two categories. The first is the responsibility r(i, k), which is the accumulated evidence that sample k should be the exemplar for sample i. The second is the availability a(i, k) which is the accumulated evidence that sample i should choose sample k to be its exemplar, and considers the values for all other samples that k should be an exemplar. In this way, exemplars are chosen by samples if they are (1) similar enough to many samples and (2) chosen by many samples to be representative of themselves.

### More formally, the responsibility of a sample k to be the exemplar of sample i is given by:

#$r(i, k) \leftarrow s(i, k) - max [ a(i, k') + s(i, k') \forall k' \neq k ]$ 

### Where s(i, k) is the similarity between samples i and k. The availability of sample k to be the exemplar of sample i is given by:

#$a(i, k) \leftarrow min [0, r(k, k) + \sum_{i'~s.t.~i' \notin \{i, k\}}{r(i', k)}]$

### To begin with, all values for r and a are set to zero, and the calculation of each iterates until convergence. As discussed above, in order to avoid numerical oscillations when updating the messages, the damping factor \lambda is introduced to iteration process:

#$r_{t+1}(i, k) = \lambda\cdot r_{t}(i, k) + (1-\lambda)\cdot r_{t+1}(i, k)$

#$a_{t+1}(i, k) = \lambda\cdot a_{t}(i, k) + (1-\lambda)\cdot a_{t+1}(i, k)$

### where t indicates the iteration times.


#####Affinity Propagation######

def compare_clusters(X,Y,method='ap',s=200):
    A = (X/np.linalg.norm(X,axis=0)).T
    #A[np.isnan(A)] = 0
    B = (Y/np.linalg.norm(Y,axis=0)).T
    #B[np.isnan(B)] = 0
    random_samples = np.zeros(A.shape[0],dtype=np.bool)
    random_samples[:min(s,A.shape[0])] = True
    np.random.shuffle(random_samples)
    A = A[random_samples]
    B = B[random_samples]
    dA = 1 - A.dot(A.T)
    dA = np.exp(-dA**2/2.)
    dB = 1 - B.dot(B.T)
    dB = np.exp(-dB**2/2.)
    del A,B
    if method == 'spectral':
        lA = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dA)
        lB = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dB)
    elif method == 'ap':
        lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
        lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
    return adjusted_mutual_info_score(lA,lB)
	
compare_clusters(X_ABC,X_DF)

#AMI (Adjusted Mutual Information) score of 0.7808506717164001
#compared with 0.34121671171724199 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]


def compare_clusters(X,Y,method='ap',s=200):
    A = (X/np.linalg.norm(X,axis=0)).T
    #A[np.isnan(A)] = 0
    B = (Y/np.linalg.norm(Y,axis=0)).T
    #B[np.isnan(B)] = 0
    random_samples = np.zeros(A.shape[0],dtype=np.bool)
    random_samples[:min(s,A.shape[0])] = True
    np.random.shuffle(random_samples)
    A = A[random_samples]
    B = B[random_samples]
    dA = 1 - A.dot(A.T)
    dA = np.exp(-dA**2/2.)
    dB = 1 - B.dot(B.T)
    dB = np.exp(-dB**2/2.)
    del A,B
    if method == 'spectral':
        lA = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dA)
        lB = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dB)
    elif method == 'ap':
        lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
        lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
    return adjusted_rand_score(lA,lB)
	
compare_clusters(X_ABC,X_DF)

#ARI (Adjusted Rand Index) score of 0.8035951598237393
#compared with 0.52307533945641171 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]

def compare_clusters(X,Y,method='ap',s=200):
    A = (X/np.linalg.norm(X,axis=0)).T
    #A[np.isnan(A)] = 0
    B = (Y/np.linalg.norm(Y,axis=0)).T
    #B[np.isnan(B)] = 0
    random_samples = np.zeros(A.shape[0],dtype=np.bool)
    random_samples[:min(s,A.shape[0])] = True
    np.random.shuffle(random_samples)
    A = A[random_samples]
    B = B[random_samples]
    dA = 1 - A.dot(A.T)
    dA = np.exp(-dA**2/2.)
    dB = 1 - B.dot(B.T)
    dB = np.exp(-dB**2/2.)
    del A,B
    if method == 'spectral':
        lA = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dA)
        lB = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dB)
    elif method == 'ap':
        lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
        lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
    return homogeneity_completeness_v_measure(lA,lB) 

compare_clusters(X_ABC,X_DF)

#Homogeneity score of 0.85811525245419984
#compared with 0.34246215681170245 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]

#Completeness score of 0.91858052600835216
#compared with 0.3624386397847103 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]

#V_Measure score of 0.88731899915614398
#compared with 0.35216733728175009 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]



#####Spectral Clustering######

## From http://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

### SpectralClustering does a low-dimension embedding of the affinity matrix between samples, followed by a KMeans in the low dimensional space. It is especially efficient if the affinity matrix is sparse and the pyamg module is installed. SpectralClustering requires the number of clusters to be specified. It works well for a small number of clusters but is not advised when using many clusters.

### For two clusters, it solves a convex relaxation of the normalised cuts problem on the similarity graph: cutting the graph in two so that the weight of the edges cut is small compared to the weights of the edges inside each cluster. This criteria is especially interesting when working on images: graph vertices are pixels, and edges of the similarity graph are a function of the gradient of the image.

### Warning: Transforming distance to well-behaved similarities:

### Note that if the values of your similarity matrix are not well distributed, e.g. with negative values or with a distance matrix rather than a similarity, the spectral problem will be singular and the problem not solvable. In which case it is advised to apply a transformation to the entries of the matrix. For instance, in the case of a signed distance matrix, is common to apply a heat kernel:

### similarity = np.exp(-beta * distance / distance.std())

### Spectral Clustering applies clustering to a projection to the normalized Laplacian.

### In practice Spectral Clustering is very useful when the structure of the individual clusters is highly non-convex or more generally when a measure of the center and spread of the cluster is not a suitable description of the complete cluster. For instance when clusters are nested circles on the 2D plan.

### If affinity is the adjacency matrix of a graph, this method can be used to find normalized graph cuts.


def compare_clusters(X,Y,method='spectral',s=200):
    A = (X/np.linalg.norm(X,axis=0)).T
    #A[np.isnan(A)] = 0
    B = (Y/np.linalg.norm(Y,axis=0)).T
    #B[np.isnan(B)] = 0
    random_samples = np.zeros(A.shape[0],dtype=np.bool)
    random_samples[:min(s,A.shape[0])] = True
    np.random.shuffle(random_samples)
    A = A[random_samples]
    B = B[random_samples]
    dA = 1 - A.dot(A.T)
    dA = np.exp(-dA**2/2.)
    dB = 1 - B.dot(B.T)
    dB = np.exp(-dB**2/2.)
    del A,B
    if method == 'spectral':
        lA = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dA)
        lB = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dB)
    elif method == 'ap':
        lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
        lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
    return adjusted_mutual_info_score(lA,lB)

compare_clusters(X_ABC,X_DF)

#AMI (Adjusted Mutual Information) score of 0.65817033887413179
#compared with 0.34121671171724199 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]


def compare_clusters(X,Y,method='spectral',s=200):
    A = (X/np.linalg.norm(X,axis=0)).T
    #A[np.isnan(A)] = 0
    B = (Y/np.linalg.norm(Y,axis=0)).T
    #B[np.isnan(B)] = 0
    random_samples = np.zeros(A.shape[0],dtype=np.bool)
    random_samples[:min(s,A.shape[0])] = True
    np.random.shuffle(random_samples)
    A = A[random_samples]
    B = B[random_samples]
    dA = 1 - A.dot(A.T)
    dA = np.exp(-dA**2/2.)
    dB = 1 - B.dot(B.T)
    dB = np.exp(-dB**2/2.)
    del A,B
    if method == 'spectral':
        lA = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dA)
        lB = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dB)
    elif method == 'ap':
        lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
        lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
    return adjusted_rand_score(lA,lB)

compare_clusters(X_ABC,X_DF)

#ARI (Adjusted Rand Index) score of 0.71532846715328469
#compared with 0.52307533945641171 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]


def compare_clusters(X,Y,method='spectral',s=200):
    A = (X/np.linalg.norm(X,axis=0)).T
    #A[np.isnan(A)] = 0
    B = (Y/np.linalg.norm(Y,axis=0)).T
    #B[np.isnan(B)] = 0
    random_samples = np.zeros(A.shape[0],dtype=np.bool)
    random_samples[:min(s,A.shape[0])] = True
    np.random.shuffle(random_samples)
    A = A[random_samples]
    B = B[random_samples]
    dA = 1 - A.dot(A.T)
    dA = np.exp(-dA**2/2.)
    dB = 1 - B.dot(B.T)
    dB = np.exp(-dB**2/2.)
    del A,B
    if method == 'spectral':
        lA = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dA)
        lB = SpectralClustering(n_clusters=2,affinity='precomputed').fit_predict(dB)
    elif method == 'ap':
        lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
        lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
    return homogeneity_completeness_v_measure(lA,lB) 

compare_clusters(X_ABC,X_DF)

#Homogeneity score of 0.66801855197380977
#compared with 0.34246215681170245 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]

#Completeness score of 0.69620167563441948
#compared with 0.3624386397847103 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]

#V_Measure score of 0.68181899934796053
#compared with 0.35216733728175009 [using the best performing KNNClassifier (precision/recall = 0.86) with the raw feature vectors]


###########CONCLUSION#############

## Although the KNN Binary Classifier (D/F vs. A/B/C SPG Score) performed well:

### 0.87 Precision, 
### 0.88 Recall, and
### 0.87 F1-Scores,

## the use of Affinity Propagation was especially helpful in identifying and using the most representative samples to perform clustering, resulting in relatively high:

### AMI (0.7808506717164001), 
### ARI (0.8035951598237393), 
### Homogeneity (0.85811525245419984), 
### Completeness (0.91858052600835216), and 
### V-Measure (0.88731899915614398) scores.