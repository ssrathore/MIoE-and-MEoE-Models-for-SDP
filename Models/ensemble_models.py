#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import metrics
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

def adaBoost(X_train,y_train,X_test,y_test):
    num_trees = 10
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=0)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    result = {
      "Accuracy": metrics.accuracy_score(y_test, y_pred_test),
      "F1_score": metrics.f1_score(y_test, y_pred_test, average='macro'),
      "AUC_score": metrics.roc_auc_score(y_test,  y_pred_test),
      "Prediction": y_pred_test,
      "MCC" : metrics.matthews_corrcoef(y_test,  y_pred_test)
      
    }
    return result


def bagging(X_train,y_train,X_test,y_test):
    seed = 8
    kfold = model_selection.KFold(n_splits = 3)
    
    #matrices
    base_cls = DecisionTreeClassifier()
  
    # no. of base classifier
    num_trees = 3

    # bagging classifier
    model = BaggingClassifier(base_estimator = base_cls,
                              n_estimators = num_trees)
  
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    result = {
      "Accuracy": metrics.accuracy_score(y_test, y_pred_test),
      "F1_score": metrics.f1_score(y_test, y_pred_test, average='macro'),
      "AUC_score": metrics.roc_auc_score(y_test,  y_pred_test),
      "Prediction": y_pred_test,
      "MCC" : metrics.matthews_corrcoef(y_test,  y_pred_test)
      
    }
    return result

