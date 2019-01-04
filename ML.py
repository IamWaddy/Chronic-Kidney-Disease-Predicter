
# # FINDING OUT THE ACUURACY OF Algorithm Used 
# Given 24 health related attributes taken in 2-month period of 400 patients, using the information of the 158 patients with complete records to predict the outcome (i.e. whether one has chronic kidney disease) of the remaining 242 patients (with missing values in their records).
# 
# ## Summary of Results
# With proper tuning of parameters using cross-validation in the training set, the Random Forest Classfier achieves an accuracy of 88.8% and an ROC AUC of 99.2%. 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import warnings
from IPython import *
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

def auc_scorer(clf, X, y, model): # function to plot the ROC curve
    if model=='RF':
        fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1])
    elif model=='SVM':
        fpr, tpr, _ = roc_curve(y, clf.decision_function(X))
    roc_auc = auc(fpr, tpr)

    plt.figure()    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve from '+model+' model (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return fpr,tpr,roc_auc






df = pd.read_csv('kidney_disease.csv')


# ## Cleaning and preprocessing of data for training a classifier



# Map text to 1/0 and do some cleaning
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'class'},inplace=True)



df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
df.drop('id',axis=1,inplace=True)




df.head()


# ## We Check the portion of rows with NaN
# - Now the data is cleaned with improper values labelled NaN. Let's see how many NaNs are there.
# - Drop all the rows with NaN values, and build a model out of this dataset (i.e. df2)




df2 = df.dropna(axis=0)
df2['class'].value_counts()


# ## Examine correlations between different features




corr_df = df2.corr()


mask = np.zeros_like(corr_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


f, ax = plt.subplots(figsize=(11, 9))


cmap = sns.diverging_palette(220, 10, as_cmap=True)


sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlations between different predictors')
plt.show()







X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,:-1], df2['class'], 
                                                    test_size = 0.33, random_state=44,
                                                    stratify= df2['class'] )




print(X_train.shape)
print(X_test.shape)





y_train.value_counts()


# ## Choosing parameters with GridSearchCV with 10-fold cross validations.
# (Suggestion for next time: try using Bayesian model selection method)


tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],
                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}]
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,scoring='f1')
clf.fit(X_train, y_train)

print("Detailed classification report:")
y_true, lr_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, lr_pred))

confusion = confusion_matrix(y_test, lr_pred)
print('Confusion Matrix:')
print(confusion)

# Determine the false positive and true positive rates
fpr,tpr,roc_auc = auc_scorer(clf, X_test, y_test, 'RF')

print('Best parameters:')
print(clf.best_params_)
clf_best = clf.best_estimator_


# ## Examine feature importance
# Since We pruned the forest (*max_depth*=2) and decrease the number of trees (*n_estimators*=8), not all features are used.


plt.figure(figsize=(12,3))
features = X_test.columns.values.tolist()
importance = clf_best.feature_importances_.tolist()
feature_series = pd.Series(data=importance,index=features)
feature_series.plot.bar()
plt.title('Feature Importance')
imp_feature=feature_series.plot.bar()





list_to_fill = X_test.columns[feature_series>0]
print(list_to_fill)


# ## Next, We examine the rest of the dataset (with missing values across the rows)
# Are there correlations between occurence of missing values in a row? The plot suggests, seems no.


corr_df = pd.isnull(df).corr()


mask = np.zeros_like(corr_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


f, ax = plt.subplots(figsize=(11, 9))


cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()



# ## Make predictions with the best model selected above
# We filled in all NaN with 0 and pass it to the trained classifier. 


df2 = df.dropna(axis=0)
no_na = df2.index.tolist()
some_na = df.drop(no_na).apply(lambda x: pd.to_numeric(x,errors='coerce'))
some_na = some_na.fillna(0) # Fill up all Nan by zero.

X_test = some_na.iloc[:,:-1]
y_test = some_na['class']
y_true = y_test
lr_pred = clf_best.predict(X_test)
print(classification_report(y_true, lr_pred))

confusion = confusion_matrix(y_test, lr_pred)
print('Confusion Matrix:')
print(confusion)

print('Accuracy: %3f' % accuracy_score(y_true, lr_pred))
# Determine the false positive and true positive rates
fpr,tpr,roc_auc = auc_scorer(clf_best, X_test, y_test, 'RF')

df2.head()

Y=df2["class"].values
#print(list_to_fill)
X=df2[list_to_fill]
X_train, X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=24)
clf= svm.SVC(kernel='linear',random_state=24)
clf.fit(X_train,y_train)
preds=clf.predict(X_test)
clf.score(X_test,y_test)
print('Accuracy: %3f' % clf.score(X_test,y_test))
fpr,tpr,roc_auc = auc_scorer(clf, X_test, y_test, 'SVM')

#We Create a New Data Frame Which contains only the importance Features Required to Make the Prediction 
#The new Data Frame is used by the CKD Predicter

df_temp = df2[['sg','al','su','bgr','sc','pot','pcv','wc','rc','dm','class']]
df3 = df_temp

df3.to_csv('kidney_disease1.csv',sep=',',encoding='utf-8',index=False)
df4= pd.read_csv('kidney_disease1.csv')
df4.head()
