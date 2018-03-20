# Import libraries 
import numpy as np 


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

def initialize_parameters(n_x, n_h, n_y):
    """
    Initializes W1, b1, W2, b2 
    
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing W1, b1, W2, b2 
    """
    
    W1 = np.random.randn(n_h, n_x) * 0.01 # shape (n_h, n_x)
    b1 = np.zeros((n_h, 1)) # shape (n_h, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01 # shape (n_y, n_h)
    b2 = np.zeros((n_y, 1)) # shape (n_y, 1) 
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Forward propagates to calculate y_hat given X and parameters 
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing parameters 
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1 
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost 
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost 
    """
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(logprobs)/m
   
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implements backward propagation 
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients of the cross-entropy loss with respect to different parameters
    """
    # Retrieve m, W1, W2, A1 and A2
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculates dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2)))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
   
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = learning_rate):
    """
    Updates parameters using the gradient descent 
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """  
    # Update rule for each parameter
    W1 = parameters["W1"] - learning_rate*grads["dW1"]
    b1 = parameters["b1"] - learning_rate*grads["db1"]
    W2 = parameters["W2"] - learning_rate*grads["dW2"]
    b2 = parameters["b2"] - learning_rate*grads["db2"]
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
 
    n_x = X_shape[0]
    n_y = Y_shape[0]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
    
        A2, cache = forward_propagation(X, parameters) # Forward propagation     
        cost = compute_cost(A2, Y, parameters) # compute cost 
        grads = backward_propagation(parameters, cache, X, Y) # Backpropagation 
        parameters = update_parameters(parameters, grads) # Gradient descent parameter update
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
    
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions
