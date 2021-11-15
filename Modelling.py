
# %% Import Libraries
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import os
import tensorboard
from datetime import datetime 
from xgboost import XGBClassifier
import xgboost as xgb
import graphviz
from sklearn import tree
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from tensorflow.keras.utils import to_categorical 
import tensorflow as tf
import numpy as np
import pickle
print(tf.config.list_physical_devices('GPU'))


# %% For alerting completion
import winsound
frequency = 600  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

# %% Tensorboard callback definition
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# %% Results from pickle file (dumping and loading operations)
everything=pd.read_pickle(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle')
print(everything)
#with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
#    pickle.dump(everything,f)
    
# %% Load Dataset - 1) Dropping header column 2) Deleting duplicate values 3) Making the new column 'Duration'
df=pd.read_csv("D:/Semester5Project/datasetnorm.csv")
df.drop(columns=["Unnamed: 0"],inplace=True)
df.drop(index=df[df.duplicated()].index,inplace=True)
df['Duration']=df['Ltime']-df['Stime']
df.drop(columns=['Ltime','Stime'],inplace=True)
cols=list(df.columns)

# %% Imputing (mean method)
imp=SimpleImputer()
df=imp.fit_transform(df)
df=pd.DataFrame(df,columns=cols)

# %% Seperating Dataset into X and y
X=df.drop(columns=['Label'])
y=df['Label']

# For clearing memory since "df" is spatially intensive
del df
gc.collect()
gc.collect()

# %% Train Test Splitting (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
del X
del y
gc.collect()

# %% Logistic Regression
model=LogisticRegression(solver='newton-cg',max_iter=300,verbose=6,n_jobs=3)
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['Logistic Regression']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['Logistic Regression']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['Logistic Regression']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['Logistic Regression']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['Logistic Regression']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['Logistic Regression']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['Logistic Regression']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['Logistic Regression']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['Logistic Regression']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['Logistic Regression']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['Logistic Regression']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['Logistic Regression']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['Logistic Regression']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['Logistic Regression']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['Logistic Regression']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['Logistic Regression']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['Logistic Regression']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['Logistic Regression']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['Logistic Regression'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Decision Tree Model Fitting
model=DecisionTreeClassifier(max_depth=9,random_state=42)
model.fit(X_train,y_train)
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['Decision Tree Classifier']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['Decision Tree Classifier']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['Decision Tree Classifier']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['Decision Tree Classifier']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['Decision Tree Classifier']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['Decision Tree Classifier']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['Decision Tree Classifier']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['Decision Tree Classifier']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['Decision Tree Classifier']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['Decision Tree Classifier']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['Decision Tree Classifier']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['Decision Tree Classifier']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['Decision Tree Classifier']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['Decision Tree Classifier']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['Decision Tree Classifier']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['Decision Tree Classifier']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['Decision Tree Classifier']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['Decision Tree Classifier']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['Decision Tree Classifier'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Random Forest Classfier
model=RandomForestClassifier(n_estimators=300,random_state=42,verbose=3,n_jobs=3)
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['Random Forest Classifier']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['Random Forest Classifier']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['Random Forest Classifier']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['Random Forest Classifier']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['Random Forest Classifier']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['Random Forest Classifier']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['Random Forest Classifier']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['Random Forest Classifier']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['Random Forest Classifier']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['Random Forest Classifier']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['Random Forest Classifier']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['Random Forest Classifier']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['Random Forest Classifier']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['Random Forest Classifier']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['Random Forest Classifier']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['Random Forest Classifier']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['Random Forest Classifier']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['Random Forest Classifier']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['Random Forest Classifier'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Naive Bayes Classifier
model=GaussianNB()
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['Naive Bayes Classifier']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['Naive Bayes Classifier']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['Naive Bayes Classifier']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['Naive Bayes Classifier']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['Naive Bayes Classifier']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['Naive Bayes Classifier']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['Naive Bayes Classifier']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['Naive Bayes Classifier']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['Naive Bayes Classifier']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['Naive Bayes Classifier']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['Naive Bayes Classifier']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['Naive Bayes Classifier']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['Naive Bayes Classifier']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['Naive Bayes Classifier']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['Naive Bayes Classifier']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['Naive Bayes Classifier']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['Naive Bayes Classifier']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['Naive Bayes Classifier']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['Naive Bayes Classifier'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)


# %% Adaboost
estimator=RandomForestClassifier(n_estimators=6,random_state=42,verbose=3,n_jobs=-1)
model=AdaBoostClassifier(base_estimator=estimator,n_estimators=100,random_state=42)
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['AdaBoost Classifier']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['AdaBoost Classifier']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['AdaBoost Classifier']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['AdaBoost Classifier']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['AdaBoost Classifier']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['AdaBoost Classifier']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['AdaBoost Classifier']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['AdaBoost Classifier']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['AdaBoost Classifier']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['AdaBoost Classifier']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['AdaBoost Classifier']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['AdaBoost Classifier']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['AdaBoost Classifier']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['AdaBoost Classifier']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['AdaBoost Classifier']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['AdaBoost Classifier']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['AdaBoost Classifier']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['AdaBoost Classifier']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['AdaBoost Classifier'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)


# %% XGBoost 
model=XGBClassifier(predictor='gpu_predictor',
                    objective='multi:softmax',
                    n_estimators=350,
                    scale_pos_weight=20,
                    max_depth=9,
                    num_class=2,
                    verbosity=3)

# %% XGBoost Fitting
model.fit(X_train,y_train,verbose=True)
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['XGBoost']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['XGBoost']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['XGBoost']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['XGBoost']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['XGBoost']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['XGBoost']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['XGBoost']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['XGBoost']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['XGBoost']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['XGBoost']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['XGBoost']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['XGBoost']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['XGBoost']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['XGBoost']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['XGBoost']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['XGBoost']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['XGBoost']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['XGBoost']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['XGBoost'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)


# %% SVM Fitting
model=SGDClassifier(loss='squared_hinge',n_jobs=3,random_state=42)
model=make_pipeline(model)
model.fit(X_train,y_train)
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Uncertainty Analysis
for x in range(20):
    print('Iteration '+str(x)+' running')
    #Shuffling X_train and X_test
    X_train=X_train.sample(frac=1)
    y_train=y_train.loc[list(X_train.index)]
    X_test=X_test.sample(frac=1)
    y_test=y_test.loc[list(X_test.index)]
    #Making predictions for both sets
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    # Appending accuracies for each iteration (Stratified KFold of 5 splits)
    a.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    b.append(np.mean(cross_val_score(model, X=X_train, y=y_train, cv=5, n_jobs=3))*100)
    train=confusion_matrix(y_train,y_pred_train)
    test=confusion_matrix(y_test,y_pred_test)
    tn1=train[0][0]
    fp1=train[0][1]
    tp1=train[1][1]
    fn1=train[1][0]
    tn2=test[0][0]
    fp2=test[0][1]
    tp2=test[1][1]
    fn2=test[1][0]
    #Appending DRs and FARs for both sets
    c.append((tp2/(tp2+fn2)))
    d.append(((fp2+fn2)/(fp2+tp2+fn2+tn2)))
    e.append((tp1/(tp1+fn1)))
    f.append(((fp1+fn1)/(fp1+tp1+fn1+tn1)))
    crtrain=classification_report(y_train,y_pred_train,output_dict=True)
    crtest=classification_report(y_test,y_pred_test,output_dict=True)
    #Appending Precision, Recall, F1 Score for both sets
    g.append(crtrain['0.0']['precision'])
    h.append(crtest['0.0']['precision'])
    i.append(crtrain['0.0']['recall'])
    j.append(crtest['0.0']['recall'])
    k.append(crtrain['0.0']['f1-score'])
    l.append(crtest['0.0']['f1-score'])
    m.append(crtrain['1.0']['precision'])
    n.append(crtest['1.0']['precision'])
    o.append(crtrain['1.0']['recall'])
    p.append(crtest['1.0']['recall'])
    q.append(crtrain['1.0']['f1-score'])
    r.append(crtest['1.0']['f1-score'])
#Saving mean and standard deviations calculated from uncertainty analysis into the dataframe loaded from pickle
everything.loc['SVM']['Training Accuracy']=str(np.mean(np.array(a)))+"±"+str(np.var(np.array(a)))
everything.loc['SVM']['Testing Accuracy']=str(np.mean(np.array(b)))+"±"+str(np.var(np.array(b)))
everything.loc['SVM']['Training DR']=str(np.mean(np.array(c)))+"±"+str(np.var(np.array(c)))
everything.loc['SVM']['Training FAR']=str(np.mean(np.array(d)))+"±"+str(np.var(np.array(d)))
everything.loc['SVM']['Testing DR']=str(np.mean(np.array(e)))+"±"+str(np.var(np.array(e)))
everything.loc['SVM']['Testing FAR']=str(np.mean(np.array(f)))+"±"+str(np.var(np.array(f)))
everything.loc['SVM']['Precision for No Attack (Train)']=str(np.mean(np.array(g)))+"±"+str(np.var(np.array(g)))
everything.loc['SVM']['Precision for No Attack (Test)']=str(np.mean(np.array(h)))+"±"+str(np.var(np.array(h)))
everything.loc['SVM']['Recall for No Attack (Train)']=str(np.mean(np.array(i)))+"±"+str(np.var(np.array(i)))
everything.loc['SVM']['Recall for No Attack (Test)']=str(np.mean(np.array(j)))+"±"+str(np.var(np.array(j)))
everything.loc['SVM']['F1 Score for No Attack (Train)']=str(np.mean(np.array(k)))+"±"+str(np.var(np.array(k)))
everything.loc['SVM']['F1 Score for No Attack (Test)']=str(np.mean(np.array(l)))+"±"+str(np.var(np.array(l)))
everything.loc['SVM']['Precision for Attack (Train)']=str(np.mean(np.array(m)))+"±"+str(np.var(np.array(m)))
everything.loc['SVM']['Precision for Attack (Test)']=str(np.mean(np.array(n)))+"±"+str(np.var(np.array(n)))
everything.loc['SVM']['Recall for Attack (Train)']=str(np.mean(np.array(o)))+"±"+str(np.var(np.array(o)))
everything.loc['SVM']['Recall for Attack (Test)']=str(np.mean(np.array(p)))+"±"+str(np.var(np.array(p)))
everything.loc['SVM']['F1 Score for Attack (Train)']=str(np.mean(np.array(q)))+"±"+str(np.var(np.array(q)))
everything.loc['SVM']['F1 Score for Attack (Test)']=str(np.mean(np.array(r)))+"±"+str(np.var(np.array(r)))
print(everything.loc['SVM'])
#Dumping new values into new pickle
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% XGBoost Feature Plot
plt.style.use('seaborn')
xgb.plot_importance(model, importance_type='gain',height=5.2)
fig = plt.gcf()
fig.set_size_inches(350, 200)

# %% Tree Plot
xgb.plot_tree(model, num_trees=0,rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(400, 500)
fig.savefig('abc.png')


# %% Keras N.N Model
def baseline_model():
    model=models.Sequential()
    model.add(layers.Dense(140,input_shape=X_train.shape,kernel_initializer='LecunNormal'))
    model.add(layers.Dropout(0.22, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.17, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.13, noise_shape=None, seed=7))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.095, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.06, noise_shape=None, seed=42))
    model.add(layers.Dense(2,activation='softmax'))
    return model

y_train = to_categorical(y_train, dtype ="uint8") 
y_test = to_categorical(y_test, dtype ="uint8") 
model=baseline_model()
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12,restore_best_weights=True)
opt = Adamax(learning_rate=0.0012)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=250,batch_size=2048,
          callbacks=[earlystop,tensorboard_callback])
y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)
y_train=np.argmax(y_train, axis=1)
y_test=np.argmax(y_test, axis=1)
y_pred_train=np.argmax(y_pred_train, axis=1)
y_pred_test=np.argmax(y_pred_test, axis=1)
trainacc=(model.evaluate(X_train,to_categorical(y_train))[1])
testacc=(model.evaluate(X_test,to_categorical(y_test))[1])
everything.loc['Neural Network']['Training Accuracy']=trainacc*100
everything.loc['Neural Network']['Testing Accuracy']=testacc*100
train=confusion_matrix(y_train,y_pred_train)
test=confusion_matrix(y_test,y_pred_test)
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['Neural Network']['Training DR']=(tp2/(tp2+fn2))
everything.loc['Neural Network']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['Neural Network']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['Neural Network']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,y_pred_train,output_dict=True)
crtest=classification_report(y_test,y_pred_test,output_dict=True)
everything.loc['Neural Network']['Precision for No Attack (Train)']=crtrain['0']['precision']
everything.loc['Neural Network']['Precision for No Attack (Test)']=crtest['0']['precision']
everything.loc['Neural Network']['Recall for No Attack (Train)']=crtrain['0']['recall']
everything.loc['Neural Network']['Recall for No Attack (Test)']=crtest['0']['recall']
everything.loc['Neural Network']['F1 Score for No Attack (Train)']=crtrain['0']['f1-score']
everything.loc['Neural Network']['F1 Score for No Attack (Test)']=crtest['0']['f1-score']
everything.loc['Neural Network']['Precision for Attack (Train)']=crtrain['1']['precision']
everything.loc['Neural Network']['Precision for Attack (Test)']=crtest['1']['precision']
everything.loc['Neural Network']['Recall for Attack (Train)']=crtrain['1']['recall']
everything.loc['Neural Network']['Recall for Attack (Test)']=crtest['1']['recall']
everything.loc['Neural Network']['F1 Score for Attack (Train)']=crtrain['1']['f1-score']
everything.loc['Neural Network']['F1 Score for Attack (Test)']=crtest['1']['f1-score']
print(everything.loc['Neural Network'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Model Redefintion for ROC-AUC Scores
model=LogisticRegression(solver='newton-cg',max_iter=300,verbose=6,n_jobs=-1)
model1=DecisionTreeClassifier(max_depth=9,random_state=42)
model2=RandomForestClassifier(n_estimators=300,random_state=42,verbose=3,n_jobs=-1)
model3=GaussianNB()
estimator=RandomForestClassifier(n_estimators=3,random_state=42,verbose=3,n_jobs=-1)
model4=AdaBoostClassifier(base_estimator=estimator,n_estimators=200,random_state=42)
model5=XGBClassifier(predictor='gpu_predictor',objective='multi:softmax',n_estimators=300,scale_pos_weight=20,max_depth=9,num_class=2,verbosity=3)
model6=SGDClassifier(loss='squared_hinge',n_jobs=3,random_state=42)

# %% ROC-AUC Score for Model 1
model.fit(X_train,y_train)
everything.loc['Logistic Regression']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
everything.loc['Logistic Regression']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test))
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for Model 2
model1.fit(X_train,y_train)
everything.loc['Decision Tree']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model1.predict_proba(X_train)[:,1])
everything.loc['Decision Tree']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model1.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test)']])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for Model 3
model2.fit(X_train,y_train)
everything.loc['Random Forest Classifier']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model2.predict_proba(X_train)[:,1])
everything.loc['Random Forest Classifier']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model2.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test)']])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for Model 4
model3.fit(X_train,y_train)
everything.loc['Naive Bayes Classifier']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model3.predict_proba(X_train)[:,1])
everything.loc['Naive Bayes Classifier']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model3.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test)']])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for Model 5
model4.fit(X_train,y_train)
everything.loc['AdaBoost Classifier']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model4.predict_proba(X_train)[:,1])
everything.loc['AdaBoost Classifier']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model4.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test)']])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for Model 6
model5.fit(X_train,y_train)
everything.loc['XGBoost']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model5.predict_proba(X_train)[:,1])
everything.loc['XGBoost']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model5.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test)']])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for Model 7
model6.fit(X_train,y_train)
everything.loc['SVM']['ROC-AUC Score (Train)']=roc_auc_score(y_train, model6.predict_proba(X_train)[:,1])
everything.loc['SVM']['ROC-AUC Score (Test)']=roc_auc_score(y_test, model6.predict_proba(X_test)[:,1])
print(everything[['ROC-AUC Score (Train)','ROC-AUC Score (Test)']])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% ROC-AUC Score for NN
def baseline_model():
    model=models.Sequential()
    model.add(layers.Dense(140,input_shape=X_train.shape,kernel_initializer='LecunNormal'))
    model.add(layers.Dropout(0.22, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.17, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.13, noise_shape=None, seed=7))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.095, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.06, noise_shape=None, seed=42))
    model.add(layers.Dense(2,activation='softmax'))
    return model
model7=baseline_model()
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12,restore_best_weights=True)
opt = Adamax(learning_rate=0.0012)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=250,batch_size=2048,
          callbacks=[earlystop,tensorboard_callback])
everything.loc['Neural Network']['ROC-AUC Score']=roc_auc_score(y_test, model7.predict_proba(X_test)[:,1])
print(everything['ROC-AUC Score'])
winsound.Beep(frequency, duration)

# %% Save the models
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\log.pickle','wb') as f:
    pickle.dump(model,f)
    
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\dt.pickle','wb') as f:
    pickle.dump(model1,f)
    
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\rfc.pickle','wb') as f:
    pickle.dump(model2,f)

with open(r'C:\Users\RAGHAV VERMA\Semester5Project\gaussiannb.pickle','wb') as f:
    pickle.dump(model3,f)
    
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\adaboost.pickle','wb') as f:
    pickle.dump(model4,f)
    
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\xgb.pickle','wb') as f:
    pickle.dump(model5,f)

with open(r'C:\Users\RAGHAV VERMA\Semester5Project\svm.pickle','wb') as f:
    pickle.dump(model6,f)
    
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\nn.pickle','wb') as f:
    pickle.dump(model7,f)
    
# %% Load the XGB Model
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\model.pickle','wb') as f:
    model=pickle.load(f)