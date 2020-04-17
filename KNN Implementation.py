"""
The Training shall start from here of KNN:
The main functions are here
"""

import statistics
def distance(target,example):
    dist = []
    for i in range(len(example)):
        dist.append(((example[i]-target)**2,i))
    return dist
    
def knn_train(example,target,y_train):
    
    """
    This will train the knn model based on the distances
    
    """
    dist_1 = []
    for j in range(len(example)):
        sum = 0
        for i in range(len(distance(target,example)[0][0])):
            sum+=distance(target,example)[j][0][i]
        dist_1.append((sum,j,y_train[j]))
    return dist_1
def knn_test(example,target,y_train,k):
    sorted_dist = sorted(knn_train(example,target,y_train))
    Usefuldist = sorted_dist[:k]

    mode = []
    for i in range(len(Usefuldist)):
        mode.append(Usefuldist[i][2])
    mode = np.asarray(mode)

    y_pred = statistics.mode(mode)
    return y_pred



def knn_pred(example, y_test,y_train,k):
    predicted = []
    for i in range(len(y_test)):
        a = knn_test(example,y_test[i],y_train,k)
        predicted.append(a)
    return predicted
    