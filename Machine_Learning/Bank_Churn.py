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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import inspect
from sklearn.metrics import precision_recall_curve
################################################################################
#Read Dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')
df.head(5)
df.describe()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

#EDA
corr_matrix = df.corr()

sn.heatmap(corr_matrix, annot=True)
plt.savefig("bank_churn_corr_matrix")
plt.show()

#Credit Card / Churn
ctab = pd.crosstab(index=df['credit_card'],columns=df['churn'])
pd.DataFrame({"m1":ctab.iloc[:,0] / ctab.iloc[:,0].sum(), "m2":ctab.iloc[:,1] / ctab.iloc[:,1].sum()})

#estimated salary / churn
df.loc[df['churn']==0,'estimated_salary'].mean()
df.loc[df['churn']==1,'estimated_salary'].mean()

df['estimated_salary'].hist(by=df['churn'])
plt.show()
# df.loc[15647311]
#credit score
df['credit_score'].describe()

plt.hist(df.loc[df['churn']==0,'credit_score'], bins=50, alpha=0.5, label="churn=0", density=True)
plt.hist(df.loc[df['churn']==1,'credit_score'], bins=50, alpha=0.5, label="churn=1", density=True)
plt.xlabel("credit score", size=14)
plt.ylabel("pct. dist", size=14)
plt.title("Credit score by Churn category")
plt.legend(loc='upper right')
plt.show()

#balance

plt.hist(df.loc[df['churn']==0,'balance'], bins=50, alpha=0.5, label="churn=0", density=True)
plt.hist(df.loc[df['churn']==1,'balance'], bins=50, alpha=0.5, label="churn=1", density=True)
plt.xlabel("balance", size=14)
plt.ylabel("pct. dist", size=14)
plt.title("balance by Churn category")
plt.legend(loc='upper right')
plt.show()

#churn rate by country
df.groupby("country")['churn'].mean()
df.groupby("gender")['churn'].mean()
df.groupby("tenure")['churn'].mean()
# plt.savefig("overlapping_histograms_with_matplotlib_Python.png")

df_transf = df.copy()
df_transf

df_transf = df_transf.set_index('customer_id')

######## Prepare for modeling ###############


df_transf.info()

# uniqueX = df_transf['gender'].unique()
# u = len(uniqueX)
# listX = [None]*u
# for i in range(0,k):
#    listX[i] = [uniqueX[i],i]
# listX

#labelencoder is good for 2 cat
labelencoder = LabelEncoder()
df_transf['gender_cat'] = labelencoder.fit_transform(df_transf['gender'])
df_transf[['gender_cat','gender']] #female = 0
df_transf = df_transf.drop(columns='gender')
# df_transf['country_cat'] = labelencoder.fit_transform(df_transf['country'])
# df_transf.groupby(['country_cat','country'])['churn'].count() #france =0, germany=1, spain=2

# creating instance of one-hot-encoder
# enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
# enc_df_transf = pd.DataFrame(enc.fit_transform(df_transf[['country']]).toarray())

# merge with main df_transf bridge_df_transf on key values
# df_transf = df_transf.join(enc_df_transf)
# df_transf.info()

# generate binary values using get_dummies
df_transf = pd.get_dummies(df_transf, columns=["country"], prefix=["country_"])
# merge with main df bridge_df on key values
df_transf.info()

X_tr,X_ts,y_tr,y_ts = train_test_split(df_transf.drop(columns='churn'),df_transf['churn'],test_size=0.3)
# inspect.getsource(train_test_split) #view function def
# train_test_split.__code__.co_varnames #view list of arguments
# train_test_split.__code__.co_argcount

######## Logistic model ###############
clf = LogisticRegressionCV(cv=10,random_state=0).fit(X_tr,y_tr)
# inspect.getmembers(clf)
#dir(clf)
# help(LogisticRegression)
# inspect.getsource(LogisticRegression)
# LogisticRegression.__code__.co_varnamess
clf.score(X_ts,y_ts)

y_ts_pred = clf.predict(X_ts)
y_ts_prob_ = clf.predict_proba(X_ts)
y_ts_prob = []
for x in range(len(y_ts_prob_)):
    y_ts_prob.append(y_ts_prob_[x][1])
    
y_df = pd.DataFrame(data={'y_ts':y_ts,'y_ts_pred':y_ts_pred,'y_ts_prob':y_ts_prob})
conf_matrix = confusion_matrix(y_ts,y_ts_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


precision, recall, thresholds = precision_recall_curve(y_ts, y_ts_prob_[:, 
1]) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = auc(recall, precision)
f1_ = 2*precision*recall/(precision+recall)
plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.plot(thresholds, f1_[: -1], "g--", label="F1")
plt.ylabel("Precision, Recall, f1")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])
f1_ = np.nan_to_num(f1_,nan=0)

precision[np.argmax(f1_)]
recall[np.argmax(f1_)]
thresholds[np.argmax(f1_)]
#Scored_ts = pd.DataFrame(clf.predict_proba(X_ts)
roc_auc_score(y_ts, y_ts_pred)