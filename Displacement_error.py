import numpy as np
import pandas as pd
# Predict error
# Average Displacement error

def ADE(pred,truth): 
    counter=0
    sum=0
    for i in range(len(pred)):
        half=int(len(pred[i])/2)
        for j in range (half):

            a = np.array((pred[i][j] , pred[i][half+j]))
            b = np.array((truth.iloc[i][j] , truth.iloc[i][half+j]))

            dist = np.linalg.norm(a-b)
            sum+=dist
            counter+=1
            #print("Distance between",a," and ",b," is: ",dist)

    return (sum/counter)

def FDE(pred,truth): 
    counter=0
    sum=0
    for i in range(len(pred)):
        half=int(len(pred[i])/2)
        last=(len(pred[i]) - 1)

        a = np.array((pred[i][half-1] , pred[i][last]))
        b = np.array((truth.iloc[i][half-1] , truth.iloc[i][last]))

        dist = np.linalg.norm(a-b)
        sum+=dist
        counter+=1
        #print("FDE Distance between",a," and ",b," is: ",dist)
            
    return (sum/counter)

