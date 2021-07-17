import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from pandas.io.pytables import Selection
from model_trainer import train_model
from sklearn.preprocessing import MinMaxScaler

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        print("The f",f)
        j=0
        for line in f:
            j+=1
           
            lined = line.strip().split("\t")
           # print("the line: ",lined)
            # try:
            line_floats = [float(i) for i in lined]

            # except ValueError:
            #   print (j) #'Line {i} is corrupt!'.format(i = index)
            #   break
            data.append(line_floats)
    return np.asarray(data)

def load_data():
    df_train=read_file('/home/murdizm/Desktop/prediction/zara1/train/crowds_zara02_train.txt')
    df_validate=read_file('/home/murdizm/Desktop/prediction/zara1/val/crowds_zara02_val.txt')
    df_test=read_file('/home/murdizm/Desktop/prediction/zara1/test/crowds_zara01.txt')

    # df_train=read_file('/home/murdizm/Desktop/prediction/eth/test/fetene.txt')
    # df_validate=read_file('/home/murdizm/Desktop/prediction/eth/test/fetene.txt')
    # df_test=read_file('/home/murdizm/Desktop/prediction/eth/test/fetene.txt')

    return df_train,df_validate,df_test


def group_id(data):
   
    unique_ids=data['Id'].unique()
    ordered_data=[]
    data_grouped=data.groupby(['Id'])
    
    for i in unique_ids:
      ordered_data.append(data_grouped.get_group(i))

    return ordered_data


def double_time_series(df,input_points,output_points): 


        input_sequence=[]
        output_sequence=[]

        
        for i in range (0,len(df)): 

            # Iterate through point coordinates from consecutive image frames
            limit=len(df[i])
            for start in range (0,limit): 

                input_sequence_points=[]
                output_sequence_points=[]
                # Check is the number of coordinates is enough to create a sequence for input data and output
                if ((start + input_points + output_points) <= limit):  

                   for k in range (input_points):
                        index= start + k 
                        temp_xy=[]
                        temp_xy.append((df[i].iloc[index]['X']))
                        temp_xy.append((df[i].iloc[index]['Y']))
                        input_sequence_points.append(temp_xy)
                
                    
                   # create a sequence of 'num_of_points' coordinates ,i.e if num_of_points is 3 , 3 points are used to create a sequence 
                   for k in range (output_points):

                        index= start + input_points + k     
                        temp_xy=[]
                        temp_xy.append(df[i].iloc[index]['X'])
                        temp_xy.append(df[i].iloc[index]['Y']) 
                        output_sequence_points.append(temp_xy)
                  
                   # Add sequences )
                   input_sequence.append(input_sequence_points)
                   output_sequence.append(output_sequence_points)
                    

        
        # change into numpy arrays (need to be transposed to be read as column in data frame) and then into dataframes
        input_sequence=np.array(input_sequence)
        output_sequence=np.array(output_sequence)


        return input_sequence,output_sequence



def divide_time_series(df,input_points,output_points,type='single'): 
      
       
        ''' create an empty list that will later be used to hold columns (x and y values). 
        The number of points determines the number of lists(number of columns,x and y coordinates)

        i.e if 3 points are used as input, we need 6 lists (later changed to array columns) for 
        x1,x2,x3,y3,y2,y1.
        The same goes for output list. If a sequence of 3 points(coordinates) are needed for prediction,
        then 3 x coordinates and 3 y coordinates are needed.'''

        input_sequence_columns=[[] for i in range(2*(input_points))]
        output_sequence_columns=[[] for i in range(2*output_points)]

        input_sequence_points=[[] for i in range(input_points)]
        output_sequence_points=[[] for i in range(output_points)]

       
        # # print("Ok",df[0].iloc[0]['Y'])

        # Iterate through Ids (different pedestrians)
      
        
        for i in range (0,len(df)): 

            # Iterate through point coordinates from consecutive image frames
            limit=len(df[i])
            for start in range (0,limit): 

                # Check is the number of coordinates is enough to create a sequence for input data and output
                if ((start + input_points + output_points) <= limit):  

                   # createindex a sequence of 'num_of_points' coordinates ,i.e if num_of_points is 3 , 3 points are used to create a sequence 
                   base_X= float(df[i].iloc[start]['X'])
                   base_Y= float(df[i].iloc[start]['Y'])

                  

                   #input_sequence_columns[0].append(df[i].iloc[start]['X'])
                   #input_sequence_columns[1].append(df[i].iloc[start]['Y'])
                   for k in range (input_points):
                        # append X and Y coordinates at j into column lists 
                        index= start + k 
                        y_column= k + input_points   # y columns start after all x columns
                        
                        # type single means coordinates are given in differnt columns but if its double then
                        # one coordinate is given as a column ==> i.e if 3 points are given
                        # single will have 6 columns while double has 3 columns 
                        temp_xy=[]
                        temp_xy.append((df[i].iloc[index]['X']))
                        temp_xy.append((df[i].iloc[index]['Y']))
                        input_sequence_points.append(temp_xy)

                        input_sequence_columns[k].append((df[i].iloc[index]['X']))
                        input_sequence_columns[y_column].append((df[i].iloc[index]['Y']))
                    
                   # create a sequence of 'num_of_points' coordinates ,i.e if num_of_points is 3 , 3 points are used to create a sequence 
                   for k in range (output_points):
                        # append X and Y coordinates at j into column lists 
                        # output prediction starts from the point after the last input point
                        index= start + input_points + k 
                        y_column= k + output_points # y columns start after all x columns
                        output_sequence_columns[k].append(df[i].iloc[index]['X'])
                        output_sequence_columns[y_column].append(df[i].iloc[index]['Y'])    


                        temp_xy=[]
                        temp_xy.append(df[i].iloc[index]['X'])
                        temp_xy.append(df[i].iloc[index]['Y']) 
                        output_sequence_points.append(temp_xy)
        
        # change into numpy arrays (need to be transposed to be read as column in data frame) and then into dataframes
        input_sequence_columns=np.array(input_sequence_columns).transpose()
        output_sequence_columns=np.array(output_sequence_columns).transpose()

        input_df=pd.DataFrame(input_sequence_columns)
        output_df=pd.DataFrame(output_sequence_columns)

        return input_df,output_df


# load_Data
array_train,array_validate,array_test=load_data()

# transform Data into DataFrames
df_train=pd.DataFrame(array_train,columns=['Frame_num','Id','X','Y'])
df_validate=pd.DataFrame(array_validate,columns=['Frame_num','Id','X','Y'])
df_test=pd.DataFrame(array_test,columns=['Frame_num','Id','X','Y'])


# Frame number does not matter since data will be provided sequentially
df_train=df_train.drop(['Frame_num'],axis=1)
df_validate=df_validate.drop(['Frame_num'],axis=1)
df_test=df_test.drop(['Frame_num'],axis=1)


# separate info based on ID
df_train_ordered=group_id(df_train)
df_validate_ordered=group_id(df_validate)
df_test_ordered=group_id(df_test)



# number of inputs and output sequences needed
num_of_inputs= 8
num_of_outputs= 8

# #creating timeline ( series of points)
# train_input,train_target=divide_time_series(df_train_ordered,num_of_inputs,num_of_outputs)
# validate_input,validate_target=divide_time_series(df_validate_ordered,num_of_inputs,num_of_outputs)
# test_input,test_target=divide_time_series(df_test_ordered,num_of_inputs,num_of_outputs)

train_input,train_target=double_time_series(df_train_ordered,num_of_inputs,num_of_outputs)
validate_input,validate_target=double_time_series(df_validate_ordered,num_of_inputs,num_of_outputs)
test_input,test_target=double_time_series(df_test_ordered,num_of_inputs,num_of_outputs)



# # Normalizaion
# scale=MinMaxScaler()
# train_input[[col for col in train_input.columns]]=scale.fit_transform(train_input[[col for col in train_input.columns]])
# validate_input[[col for col in validate_input.columns]]=scale.fit_transform(validate_input[[col for col in validate_input.columns]])
# test_input[[col for col in test_input.columns]]=scale.fit_transform(test_input[[col for col in test_input.columns]])
# # df_test[['X','Y']]=scale.fit_transform(df_test[['X','Y']])

print(train_input[0][0])
#lets train the model with linear function
degree=1
model='LSTM_2'
pre_trained=False
train_model(train_input,train_target,validate_input,validate_target,test_input,test_target,model,degree,pre_trained)