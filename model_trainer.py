from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Reshape
from keras.models import load_model,Model
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras .layers import Input
import numpy as np
import pickle
import random
from Displacement_error import ADE,FDE,prediction_displacement,ADE_double_coordinates,FDE_double_coordinates,prediction_displacement_double


# def create_hard_coded_decoder_input_model(batch_size,n_units=16):
                
#                 #initialize
#                 numberOfLSTMunits=n_units
#                 n_timesteps_in=8
#                 n_features=2

#                 # The first part is encoder
#                 encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
#                 encoder_lstm = LSTM(numberOfLSTMunits, return_state=True,  name='encoder_lstm')
#                 encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
                
#                 # initial context vector is the states of the encoder
#                 states = [state_h, state_c]
                
#                 # Set up the decoder layers
#                 # Attention: decoder receives 1 token at a time &
#                 # decoder outputs 1 token at a time 
#                 decoder_inputs = Input(shape=(1, n_features))
#                 decoder_lstm = LSTM(numberOfLSTMunits, return_sequences=True, 
#                                 return_state=True, name='decoder_lstm')
#                 decoder_dense = Dense(n_features, activation='softmax',  name='decoder_dense')

#                 all_outputs = []
#                 # Prepare decoder initial input data: just contains the START character 0
#                 # Note that we made it a constant one-hot-encoded in the model
#                 # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
#                 decoder_input_data = np.zeros((batch_size, 1, n_features))
#                 decoder_input_data[:, 0, 0] = 1 
                
#                 # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
#                 inputs = decoder_input_data
#                 # decoder will only process one time step at a time
#                 # loops for fixed number of time steps: n_timesteps_in
#                 for _ in range(n_timesteps_in):
#                                 # Run the decoder on one time step
#                                 outputs, state_h, state_c = decoder_lstm(inputs,initial_state=states)
#                                 outputs = decoder_dense(outputs)
#                                 # Store the current prediction (we will concatenate all predictions later)
#                                 all_outputs.append(outputs)
#                                 # Reinject the outputs as inputs for the next loop iteration
#                                 # as well as update the states
#                                 inputs = outputs
#                                 states = [state_h, state_c]

#                 # Concatenate all predictions such as [batch_size, timesteps, features]
#                # decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

#                 # Define and compile model 
#                 model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
#                 model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#                 return model	
# returns train, inference_encoder and inference_decoder models
def define_models(n_input=1, n_output=1, n_units=32,n_timesteps_in=8,n_features=2):
	# define training encoder
	encoder_inputs =Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')#Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='relu')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model



def reshaped(data,prediction_size,num_of_features):
   output=data.reshape(len(data),prediction_size,num_of_features)
   return output

def plotter_double(past,pred,d_truth,num_plots=6,show_together=False):
        
        past=np.array(past)
        pred=np.array(pred)
        d_truth=np.array(d_truth)
        
        for i in range(num_plots):
                n = random.randint(0,(len(pred)-1))
                past_x,past_y,pred_x,pred_y,truth_x,truth_y=[],[],[],[],[],[]
                for j in range (len(past[n])):
                   past_x.append(past[n][j][0])  
                   past_y.append(past[n][j][1])    

                for j in range (len(pred[n])):
                   pred_x.append(pred[n][j][0])  
                   pred_y.append(pred[n][j][1]) 

                   truth_x.append(d_truth[n][j][0])  
                   truth_y.append(d_truth[n][j][1])
                
                plt.scatter(past_x,past_y,c='gray')
                plt.scatter(truth_x,truth_y,c='Red')
                # #plt.plot(truth_x,truth_y,c='Red')
                plt.scatter(pred_x,pred_y,c='Green')

                if(show_together==False):
                  plt.show()

        if(show_together==True):
                 plt.show()
                

 
def plotter(past,pred,d_truth,num_plots,show_together):
        
       
        
        num_of_points=int(len(pred[0])/2)
        num_of_points_input=int(len(past.columns)/2)
        for i in range(num_plots):
                n = random.randint(0,(len(pred)-1))

                past_x,past_y,pred_x,pred_y,truth_x,truth_y=[],[],[],[],[],[]

                for j in range (num_of_points_input):

                    past_x.append(past.iloc[n][j])
                    past_y.append(past.iloc[n][num_of_points_input+j])  

                for j in range (num_of_points):

                    pred_x.append(pred[n][j])
                    pred_y.append(pred[n][num_of_points+j])


                    truth_x.append(d_truth.iloc[n][j])
                    truth_y.append(d_truth.iloc[n][num_of_points+j])
                   

                plt.scatter(past_x,past_y,c='gray')
                plt.scatter(truth_x,truth_y,c='Red')
                # #plt.plot(truth_x,truth_y,c='Red')
                plt.scatter(pred_x,pred_y,c='Green')
                #plt.plot(pred_x,pred_y,c='Green')
                #plt.scatter(pred_x[0],pred_y[0],c='blue') 
                
                # plt.xlim(-3, 10)
                # plt.ylim(-3, 10)
                if(show_together==False):
                 plt.show()
        if(show_together==True):
                 plt.show()

def show_error_double(prediction_train,prediction_test,train_target,test_target):
                # # Show error rate
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Train: ",prediction_displacement_double(train_target))
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Test: ",prediction_displacement_double(test_target))
             
        #average_displacement_error
        print("ADE ERROR RATE TEST: ", ADE_double_coordinates(prediction_test,test_target))
        #average_displacement_error
        print("ADE ERROR RATE TRAIN: ", ADE_double_coordinates(prediction_train,train_target))
        print("//////////////////////////////////////////")
        #Final_displacement_error
        print("FDE ERROR RATE TEST: ", FDE_double_coordinates(prediction_test,test_target))
        print("FDE ERROR RATE TRAIN: ", FDE_double_coordinates(prediction_train,train_target))

def show_error_single(prediction_train,prediction_test,train_target,test_target):
                # # Show error rate
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Train: ",prediction_displacement(train_target))
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Test: ",prediction_displacement(test_target))
        print("////////////////////////////////////////////////")
        #average_displacement_error
        print("ADE ERROR RATE TEST: ", ADE(prediction_test,test_target))
        #average_displacement_error
        print("ADE ERROR RATE TRAIN: ", ADE(prediction_train,train_target))
        print("//////////////////////////////////////////")
        #Final_displacement_error
        print("FDE ERROR RATE TEST: ", FDE(prediction_test,test_target))
        print("FDE ERROR RATE TRAIN: ", FDE(prediction_train,train_target))


def train_model(train_input,train_target,validatation_input,validation_target,test_input,test_target,model_kind='linear',order_of_polynomial=3,load=False): 
        

        # # Since the traning data is so small we divide the test data into validate and test data
        # test_input_updated,validatation_input,test_target,validation_target=train_test_split(test_input,test_target,test_size=0.2)
        

        if(model_kind=='linear'):
                print("Model Linear")
                #filename = 'Pre_trained_models/model_linear.sav'
                filename = 'Pre_trained_models/model_linear_2.sav'
                if (load==True):
                        # load the model from disk
                        model = pickle.load(open(filename, 'rb'))
                        
                else:
                        model= LinearRegression()
                        model.fit(train_input,train_target)
                        # save the model to disk
                        pickle.dump(model, open(filename, 'wb'))
 
                prediction_test=model.predict(test_input)
                prediction_train=model.predict(train_input)
               # print("Len of cooficients: ",model.coef_," intercepts: ",model.intercept_)

        elif(model_kind=='polynomial'):
                print("Model polynomial")
                polynomial_features=PolynomialFeatures(degree=order_of_polynomial)
                X_TRANSF=polynomial_features.fit_transform(train_input)
                #filename = 'Pre_trained_models/model_polynomial.sav'
                filename = 'Pre_trained_models/model_polynomial_2.sav'
                if (load==True):
                        # load the model from disk
                        model = pickle.load(open(filename, 'rb'))
                else:

                        model= LinearRegression()
                        model.fit(X_TRANSF,train_target)
                        # save the model to disk
                        pickle.dump(model, open(filename, 'wb'))
                
                X_test_TRANSF=polynomial_features.fit_transform(test_input)
                prediction_test=model.predict(X_test_TRANSF)
                prediction_train=model.predict(X_TRANSF)
               # print("cooficients: ",model.coef_," intercepts: ",model.intercept_)
                print("Predicted: ",prediction_test)
 
        elif(model_kind=='LSTM'):
                print("Model LSTM ")
                num_of_features=1

                # train_input=reshaped(train_input,train_input.shape[1],num_of_features)
                # #train_target=reshaped(train_target,train_target.shape[1],num_of_features)
                # validatation_input=reshaped(validatation_input,validatation_input.shape[1],num_of_features)
                # test_input=reshaped(test_input,test_input.shape[1],num_of_features)
              
     
                
                if (load==True):
                       try:
                               model = load_model("Pre_trained_models/model_LSTM_Encoder")
                       except:
                               print("No Model Has Been Saved Before!")
                               
                # else:
                #         model=Sequential()
                #         model.add(LSTM(32, input_shape=(train_input.shape[1],num_of_features),return_sequences=True))
                #         #model.add(LSTM(32,activation='relu',return_sequences=True))
                #         model.add(LSTM(16,activation='relu'))
                #         model.add(Dense(train_target.shape[1]))
                #         model.compile(loss='mse', optimizer='adam')
                #         model.fit(train_input, train_target, epochs=60, batch_size=20, verbose=1,validation_data=(validatation_input, validation_target))
                #         model.save('Pre_trained_models/model_LSTM')

                # else:
                #         model = Sequential()
                #         model.add(LSTM(16, input_shape=(train_input.shape[1],train_input.shape[2]),return_sequences=True))
                #         #model.add(RepeatVector(train_target.shape[1]))
                #         model.add(LSTM(16, return_sequences=True))
                #         #model.add(TimeDistributed(Dense(1,2)))
                #         model.add(TimeDistributed(Dense(2)))
                #         opt = Adam(learning_rate=0.005)
                #         model.compile(loss='mse', optimizer=opt)
                #         history=model.fit(train_input, train_target, epochs=30, batch_size=10, verbose=1,validation_data=(validatation_input, validation_target))
                #         model.save('Pre_trained_models/model_LSTM_Encoder')
                
                # prediction_test = model.predict(test_input)
                # prediction_train=model.predict(train_input)
                # plt.plot(history.history['loss'],c='blue')
                # plt.plot(history.history['val_loss'],c='red')
                # plt.show()

        elif(model_kind=='Encoder_Decoder'):
                print("Model Encoder_Decoder ")

                if (load==True):
                       try:
                               model_encoder = load_model("Pre_trained_models/model_Encoder_Decoder")
                       except:
                               print("No Model Has Been Saved Before!")
                else:
                        # params 
                        n_features=len(train_input[0][0])
                        n_timesteps_in=len(train_input[0])
                        input_shape=train_input[0].shape
                        n_units= 16
                        batch_size=5


                        #model,encoder,decoder=define_models()
                                # define training encoder
                        encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
                        encoder_lstm = LSTM(n_units, return_state=True,  name='encoder_lstm')
                        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
                        states = [state_h, state_c]
                        all_outputs = []

                        # Decoder layers
                        numberOfLSTMunits=16
                        decoder_inputs = Input(shape=(1, n_features), name='decoder_inputs')
                        decoder_lstm = LSTM(numberOfLSTMunits, return_sequences=True, 
                                        return_state=True, name='decoder_lstm')
                        decoder_dense = Dense(n_features, activation='softmax')
                        
                        # Prepare decoder initial input data: just contains the START character 0
                        # Note that we made it a constant one-hot-encoded in the model
                        # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
                        decoder_input_data = np.zeros((batch_size, 1, n_features))
                        #decoder_input_data = np.zeros(( 1, n_features))
                        #decoder_input_data[0, 0] = 1 
                        print("Decoder: ",decoder_input_data)
                        
                        # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
                        inputs = decoder_input_data

                        for _ in range(n_timesteps_in): 
                                # Run the decoder on one time step
                                outputs, state_h, state_c = decoder_lstm(inputs,
                                                                        initial_state=states)
                               
                                print("SIZE OF OUTPUT:     ",outputs.shape)
                                outputs = decoder_dense(outputs)
                                # Store the current prediction (we will concatenate all predictions later)
                                all_outputs.append(outputs)
                                # Reinject the outputs as inputs for the next loop iteration
                                # as well as update the states
                                inputs = outputs
                                states = [state_h, state_c]
                                print("INPUT AGAIN: ",outputs.shape)

                          # Concatenate all predictions such as [batch_size, timesteps, features]
                        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

                        # Define and compile model 
                        model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
                        opt = Adam(learning_rate=0.005)
                        model.compile(optimizer=opt, loss='mse')

                        #layer_2=Dense(n_timesteps_in*n_features,activation='relu')(encoder_outputs)
                        #output_trial=Reshape((n_timesteps_in,n_features))(layer_2)
                        print("Encoder_output",encoder_outputs.shape)


        elif(model_kind=='LSTM_2'):
                print("Model LSTM_2 ")

                if (load==True):
                       try:
                               model_encoder = load_model("Pre_trained_models/model_LSTM_Encoder_only2")
                       except:
                               print("No Model Has Been Saved Before!")
                else:
                        # params 
                        n_features=len(train_input[0][0])
                        n_timesteps_in=len(train_input[0])
                        input_shape=train_input[0].shape
                        n_units= 16


                        #model,encoder,decoder=define_models()
                                # define training encoder
                        encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
                        encoder_lstm = LSTM(n_units, return_state=True,  name='encoder_lstm')
                        layer_1 =  LSTM(n_units, return_state=True,  name='encoder_lstm')(encoder_inputs)
                        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
                        print("Encoder_output",encoder_outputs.shape)
                        layer_2=Dense(n_timesteps_in*n_features,activation='relu')(encoder_outputs)
                        output_trial=Reshape((n_timesteps_in,n_features))(layer_2)

                        
                                # Define and compile model first
                        model_encoder = Model(encoder_inputs, output_trial) 
                        opt = Adam(learning_rate=0.005)
                        model_encoder.compile(loss='mse', optimizer=opt)
                        history=model_encoder.fit(train_input,train_target,epochs=30, batch_size=5, verbose=1,validation_data=(validatation_input, validation_target))
                        model_encoder.save('Pre_trained_models/model_LSTM_Encoder_only2')
                        plt.plot(history.history['loss'],c='blue')
                        plt.plot(history.history['val_loss'],c='red')
                        plt.show()
                        plt.show()

                # encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
               
                # states = [state_h, state_c]
        

                prediction_test = model_encoder.predict(test_input)
                prediction_train=model_encoder.predict(train_input)

        # plot graphs --to show together set show_together to true

        num_of_trajectories=10  # how many trajectories will be plotted
        show_together= False 

        ## Plot
        #plotter_double(test_input,prediction_test,test_target,num_of_trajectories,show_together)
        ## show error
        #show_error_double(prediction_train,prediction_test,train_target,test_target)
        #show_error_single(prediction_train,prediction_test,train_target,test_target)



        

