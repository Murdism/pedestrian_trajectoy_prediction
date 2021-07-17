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
def prediction_displacement(pred): 
    counter=0
    sum=0
    pred=np.array(pred)
    for i in range(len(pred)):
        half=int(len(pred[i])/2)
        last=(len(pred[i]) - 1)

        a = np.array((pred[i][0] , pred[i][half]))
        b = np.array((pred[i][half-1] , pred[i][last]))

        dist = np.linalg.norm(a-b)
        sum+=dist
        counter+=1
        #print("FDE Distance between",a," and ",b," is: ",dist)
            
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


def FDE_double_coordinates(pred,truth):
          
        sum=0
        counter=0
        
        pred=np.array(pred)
        truth=np.array(truth)
        last_index= (len(pred[0])-1)

        for i in range(len(pred)):
                
            a = np.array((pred[i][last_index][0] , pred[i][last_index][1]))
            b = np.array((truth[i][last_index][0] , truth[i][last_index][1]))

            dist = np.linalg.norm(a-b)
            sum+=dist
            counter+=1

                # for j in range (len(pred[i])):
                #    pred_x.append(pred[i][j][0])  
                #    pred_y.append(pred[i][j][1]) 

                #    truth_x.append(truth[i][j][0])  
                #    truth_y.append(truth[i][j][1])
        return (sum/counter)
def ADE_double_coordinates(pred,truth):
          
        sum=0
        counter=0
        
        pred=np.array(pred)
        truth=np.array(truth)

        for i in range(len(pred)):
                
                for j in range (len(pred[i])):

                    a = np.array((pred[i][j][0] , pred[i][j][1]))
                    b = np.array((truth[i][j][0] , truth[i][j][1]))

                    dist = np.linalg.norm(a-b)
                    sum+=dist
                    counter+=1
        return (sum/counter)