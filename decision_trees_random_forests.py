import pandas as pd
import numpy as np

# Creating dataframe
loans = pd.read_csv('loan_data.csv')

'''
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   credit.policy      9578 non-null   int64  
 1   purpose            9578 non-null   object 
 2   int.rate           9578 non-null   float64
 3   installment        9578 non-null   float64
 4   log.annual.inc     9578 non-null   float64
 5   dti                9578 non-null   float64
 6   fico               9578 non-null   int64  
 7   days.with.cr.line  9578 non-null   float64
 8   revol.bal          9578 non-null   int64  
 9   revol.util         9578 non-null   float64
 10  inq.last.6mths     9578 non-null   int64  
 11  delinq.2yrs        9578 non-null   int64  
 12  pub.rec            9578 non-null   int64  
 13  not.fully.paid     9578 non-null   int64
'''

# Since the purpose column is categorical, it needs to be changed using dummy variables, so sklearn will be able to understand it

# Dataframe using dummy variables
final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True, dtype=float)

'''
Data columns (total 19 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   credit.policy               9578 non-null   int64  
 1   int.rate                    9578 non-null   float64
 2   installment                 9578 non-null   float64
 3   log.annual.inc              9578 non-null   float64
 4   dti                         9578 non-null   float64
 5   fico                        9578 non-null   int64  
 6   days.with.cr.line           9578 non-null   float64
 7   revol.bal                   9578 non-null   int64  
 8   revol.util                  9578 non-null   float64
 9   inq.last.6mths              9578 non-null   int64  
 10  delinq.2yrs                 9578 non-null   int64  
 11  pub.rec                     9578 non-null   int64  
 12  not.fully.paid              9578 non-null   int64  
 13  purpose_credit_card         9578 non-null   float64
 14  purpose_debt_consolidation  9578 non-null   float64
 15  purpose_educational         9578 non-null   float64
 16  purpose_home_improvement    9578 non-null   float64
 17  purpose_major_purchase      9578 non-null   float64
 18  purpose_small_business      9578 non-null   float64 
'''

# In the new dataframe, numerical columns were created based on the purpose column

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Decision Tree model
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

'''
              precision    recall  f1-score   support

           0       0.86      0.84      0.85      2674
           1       0.21      0.23      0.22       487

    accuracy                           0.75      3161
   macro avg       0.53      0.54      0.53      3161
weighted avg       0.76      0.75      0.75      3161
'''
'''
[[2223  416]
 [ 401  121]]
'''

# The Decision Tree model showed an accuracy of 76% 

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

'''
              precision    recall  f1-score   support

           0       0.85      1.00      0.92      2673
           1       0.52      0.02      0.04       488

    accuracy                           0.85      3161
   macro avg       0.69      0.51      0.48      3161
weighted avg       0.80      0.85      0.78      3161
'''
'''
[[2663   10]
 [ 477   11]]
'''

# The Random Forest model showed an accuracy of 80%