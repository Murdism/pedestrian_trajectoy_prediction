import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pandas.io.pytables import Selection
from model_trainer import train_model
from sklearn.preprocessing import MinMaxScaler

def load_data():
    with open('train_input.pickle', 'rb') as data:
     train_input = pickle.load(data)
    with open('validate_input.pickle', 'rb') as data:
         validate_input= pickle.load(data)
    with open('test_input.pickle', 'rb') as data:
         test_input = pickle.load(data)
    with open('train_target.pickle', 'rb') as data:
          train_target = pickle.load(data)
    with open('validate_target.pickle', 'rb') as data:
          validate_target = pickle.load(data)
    with open('test_target.pickle', 'rb') as data:
         test_target = pickle.load(data)
    return train_input,validate_input,test_input,train_target,validate_target,test_target

train_input,validate_input,test_input,train_target,validate_target,test_target=load_data()
print("Train shape: " ,len(train_input[0]))
#

#lets train the model with linear function
degree=1
model='Encoder_Decoder'
pre_trained=False
train_model(train_input,train_target,validate_input,validate_target,test_input,test_target,model,degree,pre_trained)