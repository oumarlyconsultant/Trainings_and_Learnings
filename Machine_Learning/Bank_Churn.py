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

#Import libraries
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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, auc, confusion_matrix #ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import inspect
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier

#Set display and other options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
plt.rcParams["figure.figsize"] = (12.0,10.0)
sn.set(rc={'figure.figsize':(12.0,10.0)})

#Read Dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')
print("-------------------------- Information on data -----------------------")
print(df.info())
print("\n")
print("-------------------------- Top 5 rows -----------------------")
print(df.head(5))
print("\n")
print("-------------------------- Quick summary -----------------------")
print(df.describe())
print("\n")

#Exploratory Analysis
def eda(df,var,target,histbin,distpct,save):
    """
    args
        df:
        var: 
        target: 
        histbin:
        distpct:
        save:
    """
    print("-------------------------- Quick summary of -var- --------------------------")
    overall = df[var].describe()
    where_target_0 = df.loc[df[target]==0,var].describe()
    where_target_1 = df.loc[df[target]==1,var].describe()
    print(pd.DataFrame({"overall":overall,"where_target_0":where_target_0,"where_target_1":where_target_1}))
    print("\n")
    
    print("-------------------------- Distributions of -var- by target --------------------------")
    plt.hist(df.loc[df[target]==0,var], bins=histbin, alpha=0.5, label="target=0", density=distpct)
    plt.hist(df.loc[df[target]==1,var], bins=histbin, alpha=0.5, label="target=1", density=distpct)
    plt.xlabel("-var-", size=14)
    plt.ylabel("distribution", size=14)
    plt.title("Distribution of -var- by target")
    plt.legend(loc='upper right')
    if save:
        plt.savefig("Distribution of -var- by target")
    plt.show()    
        
    print("\n")
    print("-------------------------- Correlation between -var- and target --------------------------")
    print(df[[var,target]].corr())
    print("\n")

eda(df,'credit_score','churn',50,True,False)
eda(df,'age','churn',50,True,False) 
eda(df,'balance','churn',50,True,False)  

def targetRate(df,var,target,save):
    temp = df.copy()
    temp['var_bin'] = pd.DataFrame(pd.qcut(temp[var],10,duplicates='drop'))
    chart_ = pd.DataFrame(temp.groupby('var_bin')[target].mean())
    chart_ = sn.barplot(x=chart_.index,y=chart_[target])
    if save:
        chart_.savefig("Target Rate by -var-")

targetRate(df,'credit_score','churn',False)
targetRate(df,'balance','churn',False)
targetRate(df,'age','churn',False)
#var_bin = pd.DataFrame(var_bin.groupby(by=['credit_score']).size())
#var_bin['bin'] = var_bin.index
#var_bin.reset_index()
#var_bin = var_bin
#var_bin.groupby(by=['bin'])['bin_index'].count()
#group = pd.DataFrame({'bin_index':var_bin.index,'count':var_bin.groupby(by=['bin']).size()})
#var_bin.groupby(by=['bin']).size()

#churn rate by country
# df.groupby("country")['churn'].mean()
# df.groupby("gender")['churn'].mean()
# df.groupby("tenure")['churn'].mean()

#correlation matrix
# corr_mat = df.corr()
# corr_mat.style.background_gradient(cmap='coolwarm')


def plotCorr(df):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """
    corr = df.corr()
    sn.heatmap(corr, cmap="Blues",annot=True)
    
plotCorr(df)

#Preprocess data
def encodeCategorical(df,var):
    df_transf = pd.get_dummies(df, columns=[var], prefix=[var],drop_first=True)
    return df_transf

df_transf = encodeCategorical(df,'country')
df_transf = encodeCategorical(df_transf,'gender')
  
df_transf = df_transf.set_index('customer_id')

df_transf.info()

#Train Test split
X_tr,X_ts,y_tr,y_ts = train_test_split(df_transf.drop(columns='churn'),df_transf['churn'],test_size=0.3)
# inspect.getsource(train_test_split) #view function def
# train_test_split.__code__.co_varnames #view list of arguments
# train_test_split.__code__.co_argcount

############  Logistic model with Cross Validation
clf = LogisticRegressionCV(cv=10,random_state=0).fit(X_tr,y_tr)
# inspect.getmembers(clf)
#dir(clf)
# help(LogisticRegression)
# inspect.getsource(LogisticRegression)
# LogisticRegression.__code__.co_varnamess
#clf.score(X_ts,y_ts)

predicted = clf.predict(X_ts)
accuracy_score(y_ts,predicted)
roc_auc_score(y_ts,predicted)
f1_score(y_ts,predicted)
confusion_matrix(y_ts,predicted)


predicted_prob = clf.predict_proba(X_ts)

precision, recall, thresholds = precision_recall_curve(y_ts, predicted_prob[:, 1]) 

#y_ts_pred = clf.predict(X_ts)
#y_ts_prob_ = clf.predict_proba(X_ts)
#y_ts_prob = []
#for x in range(len(y_ts_prob_)):
#    y_ts_prob.append(y_ts_prob_[x][1])
#    
#y_df = pd.DataFrame(data={'y_ts':y_ts,'y_ts_pred':y_ts_pred,'y_ts_prob':y_ts_prob})
#conf_matrix = confusion_matrix(y_ts,y_ts_pred)
#cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])

#cm_display.plot()
#plt.show()
#pr_auc = auc(recall, precision)

f1_ = 2*precision*recall/(precision+recall)
plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.plot(thresholds, f1_[: -1], "g--", label="F1")
plt.ylabel("Precision, Recall, f1")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])
#f1_ = np.nan_to_num(f1_,nan=0)

#precision[np.argmax(f1_)]
#recall[np.argmax(f1_)]
new_thresh = thresholds[np.argmax(f1_)]


####### Decision Trees
dtclf = DecisionTreeClassifier(random_state=0).fit(X_tr,y_tr)










####### Random Forest




















