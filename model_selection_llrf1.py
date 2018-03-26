import math
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import sys
import time
import os
import json

LAYER1_NODE = sys.argv[1]  # Nodes of first hidden layer
LAYER2_NODE = sys.argv[2]  # Nodes of second hidden layer
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]  # Select GPU machine 0 or 1
TEST_MODEL_NUM = sys.argv[4] # Last device for training

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the modeling

    Arguments:
    n_x -- num of parameters that we are going to analyze
    n_y -- number of classes (1)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """
    with tf.name_scope('input') as scope:
        X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
        Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")

    return X, Y

def initialize_parameters():
    """
    Initializes parameters to build a two hidden layers neural network

    Input paramters : SBPM(1) + main_dump(1) + LLRF(8) + Modulator(15) => 25 variables


    Shape: W1 : [LAYER1_NODE, 24] ([number of 1st layer neuron, number of input parameters])
           b1 : [LAYER1_NODE, 1]  ([number of 1st layer neuron, 1])
           W2 : [LAYER2_NODE, LAYER1_NODE] ([number of 2nd layer neuron, number of previous neuron])
           b2 : [LAYER2_NODE, 1]  ([number of 2nd layer neuron, 1])
           W3 : [1, LAYER2_NODE]  ([number of output paramter, number of previous neuron])
           b3 : [1, 1]   ([number of output parameter, 1])

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [int(LAYER1_NODE), 24], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [int(LAYER2_NODE), int(LAYER1_NODE)], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [1, int(LAYER2_NODE)], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))

    b1 = tf.get_variable("b1", [int(LAYER1_NODE), 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    b2 = tf.get_variable("b2", [int(LAYER2_NODE), 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    b3 = tf.get_variable("b3", [1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2,
                  "W3": W3, "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:

        Linear -> ReLU -> Linear -> ReLU -> Linear -> Distance(L2)

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of data)
    parameters -- initialize_parameters containing W1, b1, W2, b2, W3, b3

    Returns:
    Z3 -- the output of the last Linear unit
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    with tf.name_scope("Wx_b") as scope:
        Z1 = tf.matmul(W1, X) + b1   # Z1 shape : (54, None)
        A1 = tf.nn.relu(Z1)          # A1 shape : (54, None)
        Z2 = tf.matmul(W2, A1) + b2  # Z2 shape : (24, None)
        A2 = tf.nn.relu(Z2)          # A2 shape : (24, None)
        Z3 = tf.matmul(W3, A2) + b3  # A3 shape : (1, None)

    return Z3

def validattion_foward(X_val, trained_weights):

    W1 = trained_weights['W1']
    b1 = trained_weights['b1']
    W2 = trained_weights['W2']
    b2 = trained_weights['b2']
    W3 = trained_weights['W3']
    b3 = trained_weights['b3']

    Z1 = tf.matmul(W1, X_val) + b1 # Z1 shape : (54, None)
    Z2 = tf.matmul(W2, Z1) + b2  # Z2 shape : (24, None)
    Z3 = tf.matmul(W3, Z2) + b3  # Z3 shape : (1, None)

    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation that shape (1, None)
    Y -- Real values of SBPM placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    pred = Z3
    true = Y
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.square(tf.add(pred, tf.negative(true))), reduction_indices=-1)

    return cost

def compute_cost_val(predict_val, real_val):

    pred = predict_val
    true = real_val

    cost_val = tf.reduce_mean(tf.square(tf.add(pred, tf.negative(true))), reduction_indices=-1)

    return cost_val

def random_mini_batches(X, Y, mini_batch_size=256, seed=0):
    """
    Arguments:
    X -- input data, of shape (input size, number of data)
    Y -- real SBPM TMIT data, of shape(1, number of data)
    mini_batch_size - size of the mini batches

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training data
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y)
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, Y_train, X_val, Y_val, learning_rate=0.001, num_epochs=200, minibatch_size=256, print_cost=True):
    """
    Implements a two-layer tensorflow neural network: Linear -> ReLU -> Linear -> ReLU -> Linear -> Distance(L2)

    Arguments:
    X_train -- input training set, of shape (input size:24, number of training data)
    Y_train -- output training set, of shape (output size:1, number of training data)
    X_val -- input validation set, of shape (input size:24, number of validation data)
    Y_val -- ouput validation set, of shape (input size:24, number of validation data)

    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch = interval of one device
    print_cost -- True to print the cost every 10 epochs

    Returns:
    parameters -- parameters learned by the model
    """

    tf.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 2

    (n_x, m) = X_train.shape  # (n_x: input size, m : number of data in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []
    costs_val = []

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(Z3, Y)

    # Backpropagation: Use an AdamOptimizer
    with tf.name_scope("Optimization") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # merge all summaries into a single "operation" which we can execute in a session
    #tf.summary.histogram("cost", cost)
    #summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        #summary_writer = tf.summary.FileWriter(logdir='/mnt/1/ATC/modeling/log/llrf/llrf' + str(dev_num), graph=tf.get_default_graph())

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # write log
                #summary_writer.add_summary(summary, epoch * num_minibatches + i)

                epoch_cost += minibatch_cost / num_minibatches

            costs.append(epoch_cost)

            trained_weight = sess.run(parameters)
            pred_val = validattion_foward(X_val, trained_weight)
            cost_val = compute_cost_val(pred_val, Y_val)
            costs_val.append(cost_val.eval()[0]) # convert tensor to np.array

            if print_cost == True:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print("Validation cost : ", cost_val.eval()[0])

        # plot the cost

        #plt.plot(np.squeeze(costs))
        #plt.ylabel('cost')
        #plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)

        save_path = saver.save(sess, '/mnt/1/ATC/modeling/weight/llrf/llrf1_weight_' + str(TEST_MODEL_NUM), global_step=100)
        print("Model saved in file: %s" % save_path)
        print("Parameters have been trained and saved")

        return parameters, costs, costs_val

def save_weight(weights):

    weights_save = {}

    for key in weights:
        weights_save[key] = weights[key].tolist()

    with open('/mnt/1/ATC/modeling/weight/llrf/llrf1_weight_'+ str(TEST_MODEL_NUM) + '.json', 'w') as fp:
        json.dump(weights_save, fp)


### read input & target csv files
x_train = np.genfromtxt('/mnt/1/ATC/data/operation/device/llrf/llrf1_input_train.csv', dtype=np.float32, delimiter=',', skip_header=1)
y_train = np.genfromtxt('/mnt/1/ATC/data/operation/device/llrf/llrf1_target_train.csv', dtype=np.float32, delimiter=',', skip_header=1).reshape(1, -1)

x_val = np.genfromtxt('/mnt/1/ATC/data/operation/device/llrf/llrf1_input_val.csv',  dtype=np.float32, delimiter=',', skip_header=1)
y_val = np.genfromtxt('/mnt/1/ATC/data/operation/device/llrf/llrf1_target_val.csv',  dtype=np.float32, delimiter=',', skip_header=1).reshape(1, -1)

### Start training
weights, cost, cost_val = model(x_train, y_train, x_val, y_val)

### Save trained weights & costs
save_weight(weights)

pd.DataFrame(cost).to_csv('/mnt/1/ATC/modeling/cost/llrf1_cost_' + str(TEST_MODEL_NUM) + '.csv', index=False)
pd.DataFrame(cost_val).to_csv('/mnt/1/ATC/modeling/cost/llrf1_cost_val_' + str(TEST_MODEL_NUM) +'.csv', index=False)

print("Training of llrf1 is now finished!")
print("Learning time : %i seconds" % (time.clock() - start_time))
print("\n")
