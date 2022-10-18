#Importing data from kaggle
from email import header
import os
os.environ['KAGGLE_USERNAME'] = "lyoumar"
os.environ['KAGGLE_KEY'] = "0c2985e7563a90bd61a714d483151c78"


import kaggle

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('gauravtopre/bank-customer-churn-dataset', path=".")

#Unzipping file
from zipfile import ZipFile

with ZipFile('bank-customer-churn-dataset.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

################################################################################
#Setup
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

################################################################################
#Read Dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')
df.head(5)
df.describe()

#EDA
corr_matrix = df.corr()

sn.heatmap(corr_matrix, annot=True)
plt.show()

#Credit Card / Churn
ctab = pd.crosstab(index=df['credit_card'],columns=df['churn'])
pd.DataFrame({"m1":ctab.iloc[:,0] / ctab.iloc[:,0].sum(), "m2":ctab.iloc[:,1] / ctab.iloc[:,1].sum()})

#estimated salary / churn
df.loc[df['churn']==0,'estimated_salary'].mean()
df.loc[df['churn']==1,'estimated_salary'].mean()

df['estimated_salary'].hist(by=df['churn'])
plt.show()
