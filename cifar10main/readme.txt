# Project Description: 
  ModelMix is a variant of DP-SGD which can significantly improve the utility-privacy tradeoff. The optimization with ModelMix works in a manner similar to DP-SGD but in each iteration we randomly mix the previous intermidiate updates before gradient descent and use batchnorm to computate the gradient of a small group of samples. It thus applies the entropy of the convergence trajactory to amplifty the DP bound and utilize batchnorm to reduce the variance of stochatic gradient.   
  

  
# Main Components of the Code:
 ModelMix.py: the main code to run
 
 gradient_utility.py: the code to implement the SGD with ModelMix
  
 mix_utility.py: the code to mix data with public data (this part is not used in the current implementation, we include the code for completeness)
 
 resnet.py: the network architectures of various resnet that can be used to train

 ModelMixPrivacyAccount.m: the code to calculate the (eps, delta)-DP for the model trained out 
 
 
 
# How to Run the Code:
In ModelMix.py
 1. select the batch_select as the number of samples used in each BatchClipping
 2. select the start_lr as the initial learning rate
 3. select the gap_rate as the minimal coordinate-wise distance in ModelMix (corresponding to \tau in the paper)
 4. select the num_epoch as the total number of epoch to run
 5. select the noise_scale as the ratio between the magnitudge of the Gaussian noise added to the sensitivity
 6. select the clip_norm as the clipping threshold of gradient clipping
 7. run the code 

In ModelMixPrivacyAccount.m,
   fill the parameters selected above into privacyaccountant.m, and obtain the (eps, delta)-DP gaurantee for produced model. (You can compare the EpsModelMIx and EpsOrDPSGD, the eps numbers of DP-SGD with


