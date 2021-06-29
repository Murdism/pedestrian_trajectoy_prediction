from keras.engine import sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
import pickle
from Displacement_error import ADE,FDE

def reshaped(data,prediction_size,num_of_features):
   output=data.to_numpy().reshape(len(data),prediction_size,num_of_features)
   return output

def plotter(pred,d_truth,num_plots,show_together):

        num_of_points=int(len(pred[0])/2)
        for i in range(num_plots):
                pred_x,pred_y,truth_x,truth_y=[],[],[],[]
                for j in range (num_of_points):
                    pred_x.append(pred[i][j])
                    pred_y.append(pred[i][j])


                    truth_x.append(d_truth.iloc[i][j])
                    truth_y.append(d_truth.iloc[i][j])
                plt.scatter(pred_x,pred_y,c='Green')
                plt.plot(pred_x,pred_y,c='Green')

                plt.scatter(truth_x,truth_y,c='Red')
                plt.plot(truth_x,truth_y,c='Red')
                if(show_together==False):
                 plt.show()
        if(show_together==True):
                 plt.show()
def train_model(train_input,train_target,test_input,test_target,model_kind='linear',order_of_polynomial=3,load=False): 
        

        # Since the traning data is so small we divide the test data into validate and test data
        test_input_updated,validatation_input,test_target,validation_target=train_test_split(test_input,test_target,test_size=0.2)
        
        if(model_kind=='linear'):
                print("Model Linear")
                filename = 'Pre_trained_models/model_linear.sav'
                if (load==True):
                        # load the model from disk
                        model = pickle.load(open(filename, 'rb'))
                        
                else:
                        model= LinearRegression()
                        model.fit(train_input,train_target)
                        # save the model to disk
                        pickle.dump(model, open(filename, 'wb'))
 
                prediction=model.predict(test_input_updated)

        elif(model_kind=='polynomial'):
                print("Model polynomial")
                filename = 'Pre_trained_models/model_polynomial.sav'
                if (load==True):
                        # load the model from disk
                        model = pickle.load(open(filename, 'rb'))
                else:
                        polynomial_features=PolynomialFeatures(order_of_polynomial)
                        X_TRANSF=polynomial_features.fit_transform(train_input)
                        model= LinearRegression()
                        model.fit(X_TRANSF,train_target)
                        # save the model to disk
                        pickle.dump(model, open(filename, 'wb'))
                
                X_test_TRANSF=polynomial_features.fit_transform(test_input_updated)
                prediction=model.predict(X_test_TRANSF)

        elif(model_kind=='LSTM'):
                print("Model LSTM ")
                num_of_features=1

                train_input=reshaped(train_input,train_input.shape[1],num_of_features)
                #train_target=reshaped(train_target,train_target.shape[1],num_of_features)
                validatation_input=reshaped(validatation_input,validatation_input.shape[1],num_of_features)
                test_input_updated=reshaped(test_input_updated,test_input_updated.shape[1],num_of_features)
              
     
                
                if (load==True):
                        model = load_model("Pre_trained_models/model_LSTM")
                else:
                        model=Sequential()
                        model.add(LSTM(100, input_shape=(train_input.shape[1],num_of_features),return_sequences=True))
                        model.add(LSTM(100,activation='relu'))
                        model.add(Dense(train_input.shape[1]))
                        model.compile(loss='mse', optimizer='adam')
                        model.fit(train_input, train_target, epochs=60, batch_size=20, verbose=1,validation_data=(validatation_input, validation_target))
                        model.save('Pre_trained_models/model_LSTM')
                
                prediction = model.predict(test_input_updated)
        # Print predictions and ground truth
        '''for i in range (len(prediction)):
                        print(prediction[i])
                        print(test_target.iloc[i]) '''

        # plot graphs --to show together set show_together to true

        num_of_trajectories=6  # how many trajectories will be plotted
        show_together= False # 1 means separetly
        plotter(prediction,test_target,num_of_trajectories,show_together)

        # Show error rate
        #average_displacement_error
        print("ADE ERROR RATE: ", ADE(prediction,test_target))
        #Final_displacement_error
        print("FDE ERROR RATE: ", FDE(prediction,test_target))