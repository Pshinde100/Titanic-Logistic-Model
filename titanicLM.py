# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:26:36 2019

@author: Pranit
"""

# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
# pip install pandas_ml
from pandas_ml import ConfusionMatrix
import numpy as np
# from sklearn import cross_validation as cv
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO 

path="D:\\Imarticus\\Python\\titanic.csv"
titanic = pd.read_csv(path)
titanic.head(5)

# print the columns
# --------------------------------------
col = list(titanic.columns)
print(col)

# count of Rows and Columns 
# -----------------------------
titanic.shape
titanic.info
# total number of rows
# --------------------------------------
len(titanic.index)

# print the records for view
# --------------------------------------
titanic.head()


titanic.dtypes
titanic.isnull().sum()


        
titanic = titanic.drop('Name', axis = 1)
titanic = titanic.drop('Cabin', axis = 1)
titanic = titanic.drop('Ticket', axis = 1)
titanic = titanic.drop('PassengerId', axis = 1)

titanic.head()

print(titanic.SibSp.value_counts())
print(titanic.Parch.value_counts())
print(titanic.Embarked.value_counts())


# need the Y-variable to be Numeric (0 and 1) for Logistic Regression
# steps:
# 1) create a new Y-variable and initialise with 0/1 appropriately
# 2) delete the old Y value 
# -----------------------------------------------------------------

titanic['Gender'] = 0
titanic.Gender[titanic.Sex == "male"] = 1
titanic.Gender[titanic.Sex == "female"] = 0

titanic.head()

titanic = titanic.drop('Sex',axis=1)

# get the count of "YES" (1) and "NO" (0)
# --------------------------------------
titanic['Survived'].value_counts()
titanic['Survived'].value_counts()/len(titanic)

# plot the "Yes" and "No"
# --------------------------------------
sns.countplot(x='Survived', data=titanic, palette='hls')

titanic.Embarked.unique()
titanic = titanic.dropna(subset = ['Embarked'])
titanic = titanic.dropna(subset = ['Age'])

pd.get_dummies(titanic.Embarked,drop_first=True).tail(20)
pd.get_dummies(titanic.Embarked,drop_first=False).head(20)


# create the dummy variables
# for every column having more than 1 value, 
# create dummy variables
# this is done since there are characters in the factor variables
# ---------------------------------------------------------------



new_titanic = titanic.copy()
titanic.head()

# Get all the factor X-variables

factor_x = titanic.select_dtypes(exclude=["int64","float64","category"]).columns.values
print(factor_x)

# convert all X-categorical variables
# ---------------------------------
for c in factor_x:
    titanic[c] = titanic[c].astype('category',copy=False)
titanic.dtypes


# del cat_list

for var in factor_x:
    cat_list = pd.get_dummies(titanic[var], drop_first=True, prefix=var)
    # data1=bank.join(cat_list)
    new_titanic = new_titanic.join(cat_list)

new_titanic.head()


# old+dummy columns
new_col_set = new_titanic.columns
print(new_col_set)
len(new_col_set)

# data with new columns
new_titanic.head()

# get the difference of new and old columns
to_keep = list(set(new_col_set).difference(set(factor_x)))
to_keep
to_keep.sort()
to_keep
len(to_keep)


# create the final dataset with the final columns set
# ---------------------------------------------------
titanic_final = new_titanic[to_keep]
titanic_final.head()
titanic_final.columns.sort_values()
len(titanic_final.columns)

# reordering the columns
# ---------------------------------------------------
titanic_final = pd.concat(
        [titanic_final['Survived'], 
        titanic_final.drop('Survived',axis=1)],
        axis=1)

titanic_final.columns

import random as r


# split the dataset into train and test
# ---------------------------------------------------
train, test = train_test_split(titanic_final, test_size = 0.3)

print(train.shape)
print(test.shape)

total_cols = len(titanic_final.columns)
print(total_cols)

# split the train and test into X and Y variables
train_x = train.iloc[:,1:total_cols+1]
train_y = train.iloc[:,0]

train_x.iloc[0:5]

test_x  = test.iloc[:,1:total_cols+1]
test_y = test.iloc[:,0]
len(train_x)
len(test_x)


# build the base model with all columns
# -------------------------------------

# build the logistic regression model

logit_model = sm.Logit(train_y, train_x)
logit_result = logit_model.fit()
logit_result.summary2()

# do feature selection here

# cross-validation
# ----------------
# number of folds
kf = KFold(n_splits=5) 
kf.get_n_splits(train_x)
print(kf)

fold = 1
# split the training further into train and test
for train_index, test_index in kf.split(train_x):
    
    cv_train_x = train_x.iloc[train_index,]
    cv_train_y = train_y.iloc[train_index,]
    
    cv_test_x = train_x.iloc[test_index,]
    cv_test_y = train_y.iloc[test_index,]
    
    # build the model on the CV training data and predict on CV testing data
    cv_logit_model = sm.Logit(cv_train_y, cv_train_x).fit()
    cv_pdct = list(cv_logit_model.predict(cv_test_x))

    # set the default cut-off to 0.5
    # and set predictions to 0 and 1
    cv_length = len(cv_pdct)
    cv_results=list(cv_pdct).copy()
        
    for i in range(0,cv_length):
        if cv_pdct[i] <= 0.5:
            cv_results[i] = 0
        else:
            cv_results[i] = 1
    
    # accuracy score
    acc_score = accuracy_score(cv_test_y,cv_results)*100
    print('Fold={0},Accuracy={1}'.format(fold,acc_score) )
    
    fold+=1




# predict on the test set
# ---------------------------------------------------
pred_y = logit_result.predict(test_x)
y_results = list(pred_y)
pred_y
length = len(y_results)
length
# set the default cut-off to 0.5
# and set predictions to 0 and 1
for i in range(0,length):
    if y_results[i] <= 0.5:
        y_results[i] = 0
    else:
        y_results[i] = 1
        
# accuracy score
print(accuracy_score(test_y,y_results)*100)

# confusion matrix
cm=ConfusionMatrix(list(y_results),list(test_y))
print(cm)
cm.print_stats()

# Classification report : precision, recall, F-score
print(cr(test_y, y_results))

# draw the ROC curve
from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, threshold = metrics.roc_curve(test_y, y_results)
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()







###############################################################
######## Decision Tree ##########################


# gini model
# entropy model
# -------------------------------------

# Model 1) DT with gini index criteria
# -------------------------------------
clf_gini = dtc(criterion = "gini", random_state = 100, 
               max_depth=3, min_samples_leaf=5)

fit1 = clf_gini.fit(train_x, train_y)
print(fit1)

# to view the graph
from sklearn import tree
# tree visualisation
# -------------------------------------
dot_data = StringIO()

tree.export_graphviz(fit1, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True)
import pydotplus

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
from IPython.display import Image

Image(graph.create_png())

# predictions
# -------------------------------------
pred_gini = fit1.predict(test_x)
len(test_x)
pred_gini
len(test_y)
len(pred_gini)
print("Gini Accuracy is ", 
      accuracy_score(test_y,pred_gini)*100)

# create dataframe with the actual and predicted results
# -------------------------------------------------------
df_results1 = pd.DataFrame({'actual':test_y, 
                            'predicted':pred_gini})
df_results1
len(df_results1)

# another nice way to plot the results
# -------------------------------------
cm1=ConfusionMatrix(list(test_y), list(pred_gini))
cm1
# plot
# -------------------------------------
cm1
cm1.plot()
cm1.print_stats()

################################
#Model 2) DT with Entropy(Information Gain) criteria
# ----------------------------------------------------
clf_entropy=dtc(criterion="entropy", 
                random_state=100, max_depth=3, 
                min_samples_leaf=5)

fit2 = clf_entropy.fit(train_x,train_y)
print(fit2)

pred_entropy = fit2.predict(test_x)

pred_entropy
len(test_y)
len(pred_entropy)
print("Entropy Accuracy is ", 
      accuracy_score(test_y,pred_entropy)*100)

df_results2 = pd.DataFrame({'actual':test_y, 
                            'predicted':pred_entropy})
df_results2

# another nice way to plot the results
# -------------------------------------
cm2=ConfusionMatrix(list(test_y), list(pred_entropy))
cm2

# plot
# -------------------------------------
cm2.plot()
cm2.print_stats()




