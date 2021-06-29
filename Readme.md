@Murdizm June-2021

# To run code go to main :
  1. select the number of input and output sequences for the model 
    # Note : 
  ---Default
  # number of inputs and output sequences needed
  num_of_inputs= 6
  num_of_outputs= 8

  2. select the type of model you want to use
     degree- degree of polynomial (no need to change it if not using polynomial)
     pretrained - set true if you want to use pretrained models

  ---Default
  # lets train the model with linear/polynomial/LSTM function
  degree=2
  pre_trained=False
  train_model(train_input,train_target,test_input,test_target,'linear',degree,pre_trained)


  # Alot of changes can also be done from model_trainer such as number of trajectories shown in plot (by default 6 trajectories are shown in plot for visualization)--- you can also change the visibilty to show trajectories together or separetly. Moreover prediction and ground_truth values can be printed from here (this code section is commented in model_trainer.py)

  # Displacement_error holds metrics to assess the prediction of the model: ADE (average displacement error) and FDE (final displacement error) are used. The results are directly printed by model_trainer.py



  # ######### NOTE: The results show that linear works better but LSTM can be improved to give better results. ########

  # NOTE: In this experiment the difference in time between two points or positions of a pedesterian is 0.4 secs. Example: If you observe the last 8 positions of a pedestrian’s trajectory and predict the next 12 timesteps,this corresponds to an observation window of 3.2 seconds and a prediction for the next 4.8 seconds.
  
  Note: Usually having many history points is not helpful especially for linear model. linar model performs well using 4 history points to predict 4 - 6 points in the future, which means, 1.6 secs of trajectory history to predict 
  1.6  - 2.4 secs in the future.

  # The next step is to use GAN and LSTM encoders to predict 
  1. Multiple trajectories
  2. More accurate results (lower ADE and FDE)
