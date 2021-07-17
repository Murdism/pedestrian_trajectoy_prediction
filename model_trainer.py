from keras.engine import sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model,Model
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras .layers import Input
import numpy as np
import pickle
import random
from Displacement_error import ADE,FDE,prediction_displacement,ADE_double_coordinates,FDE_double_coordinates

	
# returns train, inference_encoder and inference_decoder models
def define_models(n_input=1, n_output=1, n_units=32):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
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

def plotter_double(past,pred,d_truth,num_plots,show_together=False):
        
        past=np.array(past)
        pred=np.array(pred)
        d_truth=np.array(d_truth)
        
        for i in range(6):
                past_x,past_y,pred_x,pred_y,truth_x,truth_y=[],[],[],[],[],[]
                for j in range (len(past[i])):
                   past_x.append(past[i][j][0])  
                   past_y.append(past[i][j][1])    

                for j in range (len(pred[i])):
                   pred_x.append(pred[i][j][0])  
                   pred_y.append(pred[i][j][1]) 

                   truth_x.append(d_truth[i][j][0])  
                   truth_y.append(d_truth[i][j][1])
                
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
        for i in range(len(pred)):
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
                # plt.scatter(truth_x,truth_y,c='Red')
                # #plt.plot(truth_x,truth_y,c='Red')
                # plt.scatter(pred_x,pred_y,c='Green')
                #plt.plot(pred_x,pred_y,c='Green')
                #plt.scatter(pred_x[0],pred_y[0],c='blue') 
                
                # plt.xlim(-3, 10)
                # plt.ylim(-3, 10)
                if(show_together==False):
                 plt.show()
        if(show_together==True):
                 plt.show()
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

                else:
                        model = Sequential()
                        model.add(LSTM(16, input_shape=(train_input.shape[1],train_input.shape[2]),return_sequences=True))
                        #model.add(RepeatVector(train_target.shape[1]))
                        model.add(LSTM(16, return_sequences=True))
                        #model.add(TimeDistributed(Dense(1,2)))
                        model.add(TimeDistributed(Dense(2)))
                        opt = Adam(learning_rate=0.005)
                        model.compile(loss='mse', optimizer=opt)
                        history=model.fit(train_input, train_target, epochs=30, batch_size=10, verbose=1,validation_data=(validatation_input, validation_target))
                        model.save('Pre_trained_models/model_LSTM_Encoder')
                
                prediction_test = model.predict(test_input)
                prediction_train=model.predict(train_input)
                plt.plot(history.history['loss'],c='blue')
                plt.plot(history.history['val_loss'],c='red')
                plt.show()


        elif(model_kind=='LSTM_2'):
                print("Model LSTM_2 ")
                model,encoder,decoder=define_models()
                opt = Adam(learning_rate=0.005)
                model.compile(loss='mse', optimizer=opt)
                history=model.fit(train_input,train_target)

                prediction_test = model.predict(test_input)
                prediction_train=model.predict(train_input)
                plt.plot(history.history['loss'],c='blue')
                plt.plot(history.history['val_loss'],c='red')
                plt.show()
                plt.show()
        # Print predictions and ground truth
        '''for i in range (len(prediction)):
                        print(prediction[i])
                        print(test_target.iloc[i]) '''

        # plot graphs --to show together set show_together to true

        num_of_trajectories=6  # how many trajectories will be plotted
        show_together= False 
        plotter_double(test_input,prediction_test,test_target,num_of_trajectories,show_together)
        #print("Murad:",prediction_test[0])

        
        # # Show error rate
        # print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Train: ",prediction_displacement(train_target))
        # print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Test: ",prediction_displacement(test_target))
        # print("////////////////////////////////////////////////")
        # #average_displacement_error
        # print("ADE ERROR RATE TEST: ", ADE(prediction_test,test_target))
        # #average_displacement_error
        # print("ADE ERROR RATE TRAIN: ", ADE(prediction_train,train_target))
        # print("//////////////////////////////////////////")
        # #Final_displacement_error
        # print("FDE ERROR RATE TEST: ", FDE(prediction_test,test_target))
        # print("FDE ERROR RATE TRAIN: ", FDE(prediction_train,train_target))

 

        #average_displacement_error
        print("ADE ERROR RATE TEST: ", ADE_double_coordinates(prediction_test,test_target))
        #average_displacement_error
        print("ADE ERROR RATE TRAIN: ", ADE_double_coordinates(prediction_train,train_target))
        print("//////////////////////////////////////////")
        #Final_displacement_error
        print("FDE ERROR RATE TEST: ", FDE_double_coordinates(prediction_test,test_target))
        print("FDE ERROR RATE TRAIN: ", FDE_double_coordinates(prediction_train,train_target))