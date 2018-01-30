import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


#Read in raw data (adjust absolute file paths)
schoolData = pd.read_csv('C:/Users/jkras/Desktop/All_Data_By_School_Final.csv',low_memory=False)
testScores = pd.read_csv('C:/Users/jkras/Desktop/1516testresults_masking_removed.csv', low_memory=False)
raceData = pd.read_csv('C:/Users/jkras/Desktop/Ec_Pupils_Expanded (2017 Race Compositions by School).csv',low_memory=False)


#pivot table for merging all Percent GLP for all Subjects of a given School Code (i.e., unit_code)
piv_test = pd.pivot_table(testScores, values='Percent GLP',index=['School Code'],columns='Subject')


#rename pivot table index from School Code to unit_code
piv_test.index.name = 'unit_code'

#append _GLP suffix to GLP score columns of pivot table
piv_test.columns = [str(col) + '_GLP' for col in piv_test.columns]

#reset pivot table index, so unit_code becomes a column (for later merging)
piv_alltest = piv_test.reset_index()

#merge with main data set
schoolData_alltest = schoolData.merge(piv_alltest,how='left',on='unit_code')


#list of race columns from racial composition dataset
racecols = ['Indian Male', 'Indian Female', 'Asian Male',
       'Asian Female', 'Hispanic Male', 'Hispanic Female', 'Black Male',
       'Black Female', 'White Male', 'White Female', 'Pacific Island Male',
       'Pacific Island Female', 'Two or  More Male', 'Two or  More Female',
       'Total', 'White', 'Black', 'Hispanic', 'Indian', 'Asian',
       'Pacific Island', 'Two or More', 'White_Pct', 'Majority_Minority']

#append _RACE suffix to racial composition column names
racecols_renamed = [str(col) + '_RACE' for col in racecols]

#zip together old and new racial composition column names for easy renaming
racecol_rename_dict = {i:j for i,j in zip(racecols,racecols_renamed)}

#rename columns of racial composition dataset so suffixing is evident when subsequently merged
raceData.rename(index=str, columns=racecol_rename_dict,inplace=True)

#ensure that unit_code column is of type string
raceData['unit_code'] = raceData['unit_code'].apply(str)

#append 0's to unit_code values in racial composition dataset, so no duplicates are created when merging (thanks Ranga!)
for i,j in raceData.iterrows():
    
    if len(raceData['unit_code'][i]) == 5: #only unit_code values with less than 5 digits need 0 prefixing to match up in the merge
        raceData.loc[i, 'unit_code'] = '0' + raceData['unit_code'][i]
        
#merge the racial composition dataset with the testScores and main schoolData datasets
schoolData_testRace = schoolData_alltest.merge(raceData,how='left',on='unit_code')



#subset the testScores dataset so that SBE District info can also be merged below
region_df = testScores[['School Code','SBE District']]

#rename the index of the subsetted dataframe so the unit_code column name will match during merging
region_df.rename(index=str, columns={'School Code':'unit_code'},inplace=True)

#drop all the duplicate combinations of column values to ensure no duplicated records are created when merging
region_df_unique = region_df.drop_duplicates()

#merge the subsetted dataframe of only unique column value combinations with the other merged dataframe
schoolData_testRaceRegion = schoolData_testRace.merge(region_df_unique,how='left',on='unit_code')

#check the shape of the resulting totally merged dataframe
schoolData_testRaceRegion.shape

#NB: this approach to merging throws a 'SettingWithCopyWarning'
#not sure if this matters-- I should maybe be using the df.loc[] syntax for subsetting, as on line 56 to avoid this, but I think everything still works


#display all column names if desired
#with pd.option_context('display.max_seq_items', None):
#    print(schoolData_testRaceRegion.columns)




