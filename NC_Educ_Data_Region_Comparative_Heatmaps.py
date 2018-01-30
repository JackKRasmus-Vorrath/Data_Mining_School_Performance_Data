
#aliasing completely merged dataframe from the previous merging script 'NC_Educ_Data_All_Merge.py'
all_df = schoolData_testRaceRegion

#subsetting data from the two districts of interest
all_df_NE = all_df[all_df['SBE District'] == 'Northeast Region']
all_df_W = all_df[all_df['SBE District'] == 'Western Region']

#expenditures variable list
exp_list = ['total_expense_num','salary_expense_pct','benefits_expense_pct','services_expense_pct','supplies_expense_pct','instruct_equip_exp_pct',
'other_expense_pct','lea_total_expense_num','lea_salary_expense_pct','lea_benefits_expense_pct','lea_services_expense_pct',
'lea_supplies_expense_pct','lea_instruct_equip_exp_pct','lea_other_expense_pct']

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

#concatenating lists for comparative heatmaps
hm_list = teach_list + grade_list
hm_list2 = exp_list + grade_list


#(remember to import seaborn as sns)

#NE region teacher qualification vs. performance scores heatmap
fig, ax = plt.subplots(figsize=(8,8))
hm_corr = all_df_NE[hm_list].corr()
sns.heatmap(hm_corr, ax=ax)
ax.set_title('Heatmap of (poor-performing) Northeastern Region Teacher Qualifications vs. Performance Scores');

#Western region teacher qualification vs. performance scores heatmap
fig2, ax2 = plt.subplots(figsize=(8,8))
hm_corr2 = all_df_W[hm_list].corr()
sns.heatmap(hm_corr2, ax=ax2)
ax2.set_title('Heatmap of (well-performing) Western Region Teacher Qualifications vs. Performance Scores');

#Western region expenditures vs. performance scores heatmap
fig3, ax3 = plt.subplots(figsize=(8,8))
hm_corr3 = all_df_W[hm_list2].corr()
sns.heatmap(hm_corr3, ax=ax3)
ax3.set_title('Heatmap of (well-performing) Western Region Expenses vs. Performance Scores');

#NE region expenditures vs. performance scores heatmap
fig4, ax4 = plt.subplots(figsize=(8,8))
hm_corr4 = all_df_NE[hm_list2].corr()
sns.heatmap(hm_corr4, ax=ax4)
ax4.set_title('Heatmap of (poor-performing) Northeastern Region Expenses vs. Performance Scores');