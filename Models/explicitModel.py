import numpy as np
from collections import Counter
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans, splitting_type

def Explicit_ME(X_train,y_train,X_test,y_test,experts,gate_classifier):

    initial_centers = kmeans_plusplus_initializer(X_train, 1).initialize();
    xmeans_instance = xmeans(X_train, initial_centers);
    xmeans_instance.process();
    clusters = xmeans_instance.get_clusters();
    temp_clustX = []
    temp_clustY = []
    final_clustX = []
    final_clustY = []

    arrX= np.array(X_train)
    arrY= np.array(y_train)
    
    for i in range(len(clusters)):
        tempx = []
        tempy = []
        for j in range(len(clusters[i])):
            tempx.append(arrX[clusters[i][j]])
            tempy.append(arrY[clusters[i][j]])
        temp_clustX.append(tempx)
        temp_clustY.append(tempy)
    
    #filtering clusters having only one class 
    for i in range(len(temp_clustX)):
        counter1 = Counter(temp_clustY[i])
        if(len(counter1)>1):
            final_clustX.append(temp_clustX[i])
            final_clustY.append(temp_clustY[i])
            
    bestLocalExperts = []
    pred_y_train_clust = []
    pred_y_test_clust = []
    X_train_new = X_train
    X_test_new = X_test
    count = 0
    totalClusters = len(final_clustX)
    
    for i in range(totalClusters):
        clustX = final_clustX[i]
        clustY = final_clustY[i]
        #index of best classifer by default 0
        best_expert = 0
        best_score = -2
        for j in range(len(experts)):
            local_result = experts[j](clustX,clustY,clustX,clustY)
            if(local_result["MCC"] > best_score):
                best_expert = j
                best_score = local_result["MCC"]

        bestLocalExperts.append(best_expert)
      
    #taking output of train and test data from local expert 
    for i in range(totalClusters):
        train_result = experts[bestLocalExperts[i]](final_clustX[i],final_clustY[i],X_train,y_train)
        test_result =  experts[bestLocalExperts[i]](final_clustX[i],final_clustY[i],X_test,y_test)
        pred_y_train_clust.append(train_result["Prediction"])
        pred_y_test_clust.append(test_result["Prediction"])


    #adding outputs of local as features to gate
    for data in X_train_new:
        for i in range(totalClusters):
            np.append(data,pred_y_train_clust[i][count])
        count = count + 1
    count = 0
    for data in X_test_new:
        for i in range(totalClusters):
            np.append(data,pred_y_test_clust[i][count])
        count = count + 1
    
    
    #training gate
    res =  gate_classifier(X_train_new,y_train,X_test_new,y_test)
    return res


