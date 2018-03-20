# Neural network implementation
Implements neural network with one hidden layer for binary classification  
Activation function for hidden layer is tanh()  
Activation function for output layer is sigmoid()  
Code based on materials from Coursera Deep Learning sequence

# Functions  
To learn parameters:  
nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False)

To make predictions based on learned parameters  
predict(parameters, X) 

n_h: number of nodes in the hidden layer   
X - input features, array of dimension (#_features, #_training_examples)   
Y - output, array of dimension (1, label_of_training_examples)   
print_cost: if True, print cost every 1000 iterations 
