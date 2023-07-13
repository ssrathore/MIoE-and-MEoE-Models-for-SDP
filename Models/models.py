import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sn
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans, splitting_type
from imblearn.over_sampling import SMOTE



def DecisionTree(X_train,y_train,X_test,y_test):
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "random").fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #matrices
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test,  y_pred)
#     plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#     plt.legend(loc=4)
#     plt.show()
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred, average='macro'))
    print("Auc score: ", auc)



def SVM(X_train,y_train,X_test,y_test):
    clf = svm.SVC().fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #matrices
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test,  y_pred)
#     plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#     plt.legend(loc=4)
#     plt.show()
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred, average='macro'))
    print("Auc score: ", auc)




def Logistic(X_train,y_train,X_test,y_test):
    clf = LogisticRegression(max_iter=100,penalty = 'none').fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #matrices
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test,  y_pred)
#     plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#     plt.legend(loc=4)
#     plt.show()
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred, average='macro'))
    print("Auc score: ", auc)




def Implicit_ME(X_train,y_train,X_test,y_test,clustSize):
    arr_X_train = np.array_split(X_train, clustSize)
    arr_y_train = np.array_split(y_train, clustSize)
    bestLocalExperts = []
    pred_y_train_clust = []
    pred_y_test_clust = []
    X_train_new = X_train
    X_test_new = X_test
    count = 0
    
    for i in range(clustSize):
#         clust_X_train, clust_X_test, clust_y_train, clust_y_test = train_test_split(arr_X_train[i],arr_y_train[i], test_size=0.2,random_state=1)
        clust_X = arr_X_train[i]
        clust_y = arr_y_train[i]
        dt = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(clust_X,clust_y)
        pred1 = dt.predict(clust_X)
        score1 = f1_score(clust_y, pred1, average='macro')

        svmClf = svm.SVC().fit(clust_X,clust_y)
        pred2 = svmClf.predict(clust_X)
        score2 = f1_score(clust_y, pred2, average='macro')

        logClf = LogisticRegression(max_iter=10000).fit(clust_X,clust_y)
        pred3 = logClf.predict(clust_X)
        score3 = f1_score(clust_y, pred3, average='macro')
        
        #comparison
        if(score2 > score1 and score2 > score3):
            bestLocalExperts.append('svm')
        elif(score3 > score1 and score3 > score2):
            bestLocalExperts.append('log')
        else:
            bestLocalExperts.append('dt')
    
    #taking output of train and test data from local expert 
    for i in range(clustSize):
        if(bestLocalExperts[i] == 'svm'):
            clf = svm.SVC().fit(arr_X_train[i],arr_y_train[i])
            pred_y_train_clust.append(clf.predict(X_train))
            pred_y_test_clust.append(clf.predict(X_test))
        elif(bestLocalExperts[i] == 'log'):
            clf = LogisticRegression(max_iter=10000).fit(arr_X_train[i],arr_y_train[i])
            pred_y_train_clust.append(clf.predict(X_train))
            pred_y_test_clust.append(clf.predict(X_test))
        else:
            clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(arr_X_train[i],arr_y_train[i])
            pred_y_train_clust.append(clf.predict(X_train))
            pred_y_test_clust.append(clf.predict(X_test))
    
    #adding outputs of local as features to gate
    for data in X_train_new:
        for i in range(clustSize):
            np.append(data,pred_y_train_clust[i][count])
        count = count + 1
    count = 0
    for data in X_test_new:
        for i in range(clustSize):
            np.append(data,pred_y_test_clust[i][count])
        count = count + 1
    
    
    #training gate
    gateDT = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(X_train_new,y_train)
    gateSVM = svm.SVC(probability=True).fit(X_train_new,y_train)
    gateLOG = LogisticRegression(max_iter=10000).fit(X_train_new,y_train)
    voting_clf = VotingClassifier(
    estimators=[('DecisionTree',gateDT), ('SVM',gateSVM),('Logistic',gateLOG)],voting='soft')
    voting_clf.fit(X_train_new, y_train)
    final_predictions = voting_clf.predict(X_test_new)
    print("Accuracy of Mixture of Experts:",metrics.accuracy_score(y_test, final_predictions))
    print("F1 score", f1_score(y_test,final_predictions, average='macro'))
    print("AUC score",  metrics.roc_auc_score(y_test,  final_predictions))

    
def Explicit_ME(X_train,y_train,X_test,y_test):
    oversample = SMOTE()
    Xbalance, ybalance = oversample.fit_resample(X_train, y_train)
    initial_centers = kmeans_plusplus_initializer(Xbalance, 1).initialize();
    xmeans_instance = xmeans(Xbalance, initial_centers);
    xmeans_instance.process();
    clusters = xmeans_instance.get_clusters();
    clustX = []
    clusty = []
    final_clustX = []
    final_clustY = []

    arrX= np.array(Xbalance)
    arrY= np.array(ybalance)
    for i in range(len(clusters)):
        tempx = []
        tempy = []
        for j in range(len(clusters[i])):
            tempx.append(arrX[clusters[i][j]])
            tempy.append(arrY[clusters[i][j]])
        clustX.append(tempx)
        clusty.append(tempy)
    
    for i in range(len(clustX)):
        counter1 = Counter(clusty[i])
        if(len(counter1)>1):
            final_clustX.append(clustX[i])
            final_clustY.append(clusty[i])
            
    bestLocalExperts = []
    pred_y_train_clust = []
    pred_y_test_clust = []
    X_train_new = X_train
    X_test_new = X_test
    count = 0
    clustSize = len(final_clustX)
    
    for i in range(clustSize):
#         clust_X_train, clust_X_test, clust_y_train, clust_y_test = train_test_split(arr_X_train[i],arr_y_train[i], test_size=0.2,random_state=1)
        clust_X = final_clustX[i]
        clust_y = final_clustY[i]
        dt = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(clust_X,clust_y)
        pred1 = dt.predict(clust_X)
        score1 = f1_score(clust_y, pred1, average='macro')

        svmClf = svm.SVC().fit(clust_X,clust_y)
        pred2 = svmClf.predict(clust_X)
        score2 = f1_score(clust_y, pred2, average='macro')

        logClf = LogisticRegression(max_iter=10000).fit(clust_X,clust_y)
        pred3 = logClf.predict(clust_X)
        score3 = f1_score(clust_y, pred3, average='macro')
        
        #comparison
        if(score2 > score1 and score2 > score3):
            bestLocalExperts.append('svm')
        elif(score3 > score1 and score3 > score2):
            bestLocalExperts.append('log')
        else:
            bestLocalExperts.append('dt')
    
    #taking output of train and test data from local expert 
    for i in range(clustSize):
        if(bestLocalExperts[i] == 'svm'):
            clf = svm.SVC().fit(final_clustX[i],final_clustY[i])
            pred_y_train_clust.append(clf.predict(X_train))
            pred_y_test_clust.append(clf.predict(X_test))
        elif(bestLocalExperts[i] == 'log'):
            clf = LogisticRegression(max_iter=10000).fit(final_clustX[i],final_clustY[i])
            pred_y_train_clust.append(clf.predict(X_train))
            pred_y_test_clust.append(clf.predict(X_test))
        else:
            clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(final_clustX[i],final_clustY[i])
            pred_y_train_clust.append(clf.predict(X_train))
            pred_y_test_clust.append(clf.predict(X_test))
    
    #adding outputs of local as features to gate
    for data in X_train_new:
        for i in range(clustSize):
            np.append(data,pred_y_train_clust[i][count])
        count = count + 1
    count = 0
    for data in X_test_new:
        for i in range(clustSize):
            np.append(data,pred_y_test_clust[i][count])
        count = count + 1
    
    
    #training gate
    gateDT = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(X_train_new,y_train)
    gateSVM = svm.SVC(probability=True).fit(X_train_new,y_train)
    gateLOG = LogisticRegression(max_iter=10000).fit(X_train_new,y_train)
    voting_clf = VotingClassifier(
    estimators=[('DecisionTree',gateDT), ('SVM',gateSVM),('Logistic',gateLOG)],voting='soft')
    voting_clf.fit(X_train_new, y_train)
    final_predictions = voting_clf.predict(X_test_new)
    print("Accuracy of Mixture of Experts:",metrics.accuracy_score(y_test, final_predictions))
    print("F1 score", f1_score(y_test,final_predictions, average='macro'))
    print("AUC score",  metrics.roc_auc_score(y_test,  final_predictions))


