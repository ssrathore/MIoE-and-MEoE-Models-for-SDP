import numpy as np
from collections import Counter
def Implicit_ME(X_train,y_train,X_test,y_test,totalClusters,experts,gate_classifier):
 
    arr_X_train = np.array_split(X_train, totalClusters)
    arr_y_train = np.array_split(y_train, totalClusters)
    bestLocalExperts = []
    pred_y_train_clust = []
    pred_y_test_clust = []
    X_train_new = X_train
    X_test_new = X_test
    count = 0
    
    for i in range(totalClusters):
        clustX = arr_X_train[i]
        clustY = arr_y_train[i]
        #index of best classifer by default 0
        best_expert = 0
        best_score = -2
        counter1 = Counter(clustY)
        if(len(counter1)>1):
            for j in range(len(experts)):
                local_result = experts[j](clustX,clustY,clustX,clustY)
                if(local_result["MCC"] > best_score):
                    best_expert = j
                    best_score = local_result["MCC"]

        bestLocalExperts.append(best_expert)
            
    #taking output of train and test data from local expert 
    for i in range(totalClusters):
        train_result = experts[bestLocalExperts[i]](arr_X_train[i],arr_y_train[i],X_train,y_train)
        test_result =  experts[bestLocalExperts[i]](arr_X_train[i],arr_y_train[i],X_test,y_test)
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
    
    
    res =  gate_classifier(X_train_new,y_train,X_test_new,y_test)
    return res
    
    