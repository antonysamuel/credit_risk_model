#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:03:14 2024

@author: sam
"""

import matplotlib as plt
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Loading datasets
a1 = pd.read_excel('case_study1.xlsx')
a2 = pd.read_excel('case_study2.xlsx')

df1 = a1.copy()
df2 = a2.copy()


df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]


df1_null_columns = []

for c in df1.columns:
    if df1.loc[df1[c] == -99999].shape[0] > 0:
        df1_null_columns.append(c)



columns_to_remove = []


for col in df2.columns:
    if df2.loc[df2[col] == -99999].shape[0] > 10000:
        columns_to_remove.append(col)
        


df2 = df2.drop(columns = columns_to_remove, axis = 1)    


for col in df2.columns:
    df2 = df2.loc[df2[col] != -99999]
    

df2.isna().sum()
df1.isna().sum()




for i in df1.columns:
    if i in df2.columns:
        print(i)
        
        
df = pd.merge(df1, df2, how='inner', left_on = ['PROSPECTID'] ,  right_on=['PROSPECTID'])

df.info()
df.isna().sum().sum()


#Check how many columns are categorical
cat_columns = []
for c in df.columns:
    if df[c].dtype == 'object':
        cat_columns.append(c)
        print(c)

'''
MARITALSTATUS
EDUCATION
GENDER
last_prod_enq2
first_prod_enq2
Approved_Flag
'''


df['MARITALSTATUS'].value_counts()
df['EDUCATION'].value_counts()
df['last_prod_enq2'].value_counts()

## Are these associated with target?
'''
label encoding
contigency table
'''


## MARITIAL STATUS VS appoved_flag are these associted?
'''
1. H0: Null hypothesis
    Not associated

2. H1: Alternate hypothesis
    Associated

3. Alpha (threshold of how much it can be wrong <5% or 0.05 usually>)
    Significance Level
    Strictness level
    
    Less risky = alpha more
    More risky = alpha less {vaccine manufacture etc.}
    
4. Confidence interval
    = 1 - alpha

5. Calculate evidence against H0
    p-value
    Calculated using tests.
    T-test, Chi-square, Anova
    Degree of freedom
    
6. Compare p value with alpha
    p-value < alpha => reject H0
    p-value > alpha => fail to reject H0


'''


'''
Tests:
    Chi-square => Categorical vs Categorical => Maritial status vs Approved_Flag
    T-Test => Cat. vs Num. => (2 Categories) 
    Anova => Cat vs Num =>     (>=3 Categories) Age vs Approved (Approved flags has 4 <more than 2> categories p1, p2, p3, p4)

'''


# Chi-square test
for i in cat_columns[:-1]:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)

# Since all categorical features have pval < 0.05, we will accept all.


# VIF for numerical columns
numerical_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
        numerical_columns.append(i)
        
        
'''
#Multicollinearity vs Corelation

multicollinearity => Predictability of each features by other features.
corelation => Degree of association or relation between two features -1 to 1 , is specific to linear relationship between columns

In convex functions (y = x^2 + 5 or y = x^2 etc) corelation gives misleading values
'''

'''
VIF -> Variance inflation factor
used to identify multicollinearity amoung IVs
Takes R-sq value for each IV and eliminate if crosses a threshold

VIF = 1 / (1 - R^2) for every feature

VIF RANGE -> 1- INF.

VIF = 1 -> NO multicollinearity
VIF = 1 - 5 -> low multicollinearity
VIF = 5 - 10 : moderate 
VIF > 50 : HIGH MULTIcollinearity
'''


#VIF sequentially check

vif_data = df[numerical_columns]
total_columns = vif_data.shape[1]
columns_to_keep = []
columns_index = 0

for i in range(0, total_columns):
    
    vif_value = variance_inflation_factor(vif_data, columns_index)
    print(columns_index, '---', vif_value)
    print('----------------------')
    
    if vif_value <= 6:
        columns_to_keep.append(numerical_columns[i])
        columns_index += 1
    else:
        vif_data = vif_data.drop([numerical_columns[i]], axis = 1)
        
        
# Check Anova for columns_to_kept

from scipy.stats import f_oneway

columns_to_keep_numerical = []

for i in columns_to_keep:
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    group_p1 = [value for value, group in zip(a, b) if group == 'P1']    
    group_p2 = [value for value, group in zip(a, b) if group == 'P2']    
    group_p3 = [value for value, group in zip(a, b) if group == 'P3']    
    group_p4 = [value for value, group in zip(a, b) if group == 'P4']

    f_statistics, p_value = f_oneway(group_p1, group_p2, group_p3, group_p4) 
    
    if p_value <= 0.05:
        columns_to_keep_numerical.append(i)
        
## Feature selection is done for categorical and num. features.


'''
## Label encoding for the categorical features 
cat_columns
[MARITALSTATUS
EDUCATION
GENDER
last_prod_enq2
first_prod_enq2
Approved_Flag]
'''

df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()


'''
ORDINAL FEATURES : EDUCATION
SSC : 1
12th : 2
Graduate : 3
Under-graduate : 3
Post-graduate : 4
Others : 1
Professional : 3
'''


df.loc[df['EDUCATION'] == 'SSC', ['EDUCATION']] = 1

df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']] = 1

df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']] = 2 

df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']] = 3

df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']] = 3

df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']] = 4

df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']] = 3


df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)

#Listing all final features 
features = columns_to_keep_numerical + ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'], dtype=int) ##onehot encoding

df_encoded.info()


'''
## Machine Learning Model fitting........................................................
'''

# Data Processing

# 1. Random Forest.

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1)

y.value_counts()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

y_train.value_counts()

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators= 200, random_state= 42)

rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
precision, recall, f1_score , _ = precision_recall_fscore_support(y_test, y_pred)

for i,v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 score: {f1_score[i]}")
    
    
    
# 2 XG Boost

import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder


y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



xgb_classifier = xgb.XGBClassifier(objective = 'multi:softmax', num_classes = 4)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

xgb_classifier.fit(x_train, y_train)

y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
precision, recall, f1_score , _ = precision_recall_fscore_support(y_test, y_pred)

for i,v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 score: {f1_score[i]}")
    
    
# 3 Descition tree classifier

from sklearn.tree import DecisionTreeClassifier


y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1)


dt_classifier = DecisionTreeClassifier(max_depth = 20, min_samples_split=10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

dt_classifier.fit(x_train, y_train)

y_pred = dt_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
precision, recall, f1_score , _ = precision_recall_fscore_support(y_test, y_pred)

for i,v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 score: {f1_score[i]}")
    
    
'''
XG boost is better with 77% acc    
Further finetune

HP tuning
Feature Engg: Scaling, Feature Engg, graphs 
'''
    