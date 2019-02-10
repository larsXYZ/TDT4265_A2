import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm

def shuffle_data(X,Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

def should_early_stop(validation_loss, num_steps=3):
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing)

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx)

  train_size = int(dataset_size*(1-val_percentage))
  idx_train = idx[:train_size]
  idx_val = idx[train_size:]
  X_train, Y_train = X[idx_train], Y[idx_train]
  X_val, Y_val = X[idx_val], Y[idx_val]
  return X_train, Y_train, X_val, Y_val

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def weight_initialization(input_unit,output_unit):

    if (smart_weight_initialization):
        weight_shape = (output_unit,input_unit)
        return np.random.normal(0,1/input_unit,weight_shape)
    else:
        weight_shape = (output_unit,input_unit)
        return np.random.uniform(-1,1,weight_shape)

def check_gradient(X, targets, w, epsilon, computed_gradient, layer):
    print("Checking gradient...")
    print('Shape', w.shape)
    dw = np.copy(computed_gradient)
    for k in range(10): #Selects only a few weights    #w.shape[0]
        for j in range(64): #Selects only a few weights   #w.shape[1]
            new_weight1, new_weight2 = np.copy(w), np.copy(w)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            loss1 = cross_entropy_loss_check(X, targets, new_weight1, layer)
            loss2 = cross_entropy_loss_check(X, targets, new_weight2, layer)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)

    maximum_absolute_difference = abs(computed_gradient-dw).max()
    print('maximum_absolute_difference:',maximum_absolute_difference)
    assert maximum_absolute_difference <= epsilon**2, "Absolute error was: {}".format(maximum_absolute_difference)

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def improved_sigmoid(a):
    return 1.7157*np.tanh(2/3*a)

def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def forward_output(hidden_layer, w_kj):
    a = hidden_layer.dot(w_kj.T)
    return softmax(a)

def forward_hidden(X, w_ji):
    a = X.dot(w_ji.T)

    if (use_improved_sigmoid):
        return improved_sigmoid(a)
    else:
        return sigmoid(a)

def calculate_accuracy(X, targets, w_ji, w_kj):
    hidden_layer = forward_hidden(X, w_ji)
    output = forward_output(hidden_layer, w_kj)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss_check(X, targets, w, layer):
    if layer=='output':
        output = forward_output(X, w)
    else:
        output = forward_hidden(X, w)
    assert output.shape == targets.shape
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()

def cross_entropy_loss(X, targets, w_ji, w_kj):
    hidden_layer = forward_hidden(X, w_ji)
    output = forward_output(hidden_layer, w_kj)
    assert output.shape == targets.shape
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()

def gradient_descent(hidden_layer, targets, X_batch, w_kj, w_ji, learning_rate, should_check_gradient, w_kj_vel, w_ji_vel):

    #Gradient descent for weights between hidden layer and output layer
    normalization_factor = hidden_layer.shape[0] * targets.shape[1] # batch_size * num_classes
    outputs = forward_output(hidden_layer, w_kj)
    delta_k = - (targets - outputs)

    dw_kj = delta_k.T.dot(hidden_layer)
    dw_kj = dw_kj / normalization_factor # Normalize gradient equally as loss normalization
    assert dw_kj.shape == w_kj.shape, "dw_kj shape was: {}. Expected: {}".format(dw_kj.shape, w_kj.shape)

    if should_check_gradient:
        print('gradient_descent')
        check_gradient(hidden_layer, targets, w_kj, 1e-2,  dw_kj, 'output')

    #Gradient descent for weights between input layer and hidden layer
    if (use_improved_sigmoid):
        z = X_batch.dot(w_ji.T)
        dz = 1.14393/np.cosh(2/3*z)
    else:
        hidden_layer = forward_hidden(X_batch,w_ji)
        dz = hidden_layer*(1-hidden_layer)
    
    delta_j = dz*delta_k.dot(w_kj)

    dw_ji = delta_j.T.dot(X_batch)
    dw_ji = dw_ji / (normalization_factor)

    assert dw_ji.shape == w_ji.shape, "dw_ji shape was: {}. Expected: {}".format(dw_ji.shape, w_ji.shape)

    if should_check_gradient:
        print('backpropagation')
        check_gradient(X_batch, hidden_layer, w_ji, 1e-2,  dw_ji, 'hidden')

    #Updating weights
    if (use_momentum):
        w_kj_vel = momentum_coeff*w_kj_vel + learning_rate*dw_kj
        w_ji_vel = momentum_coeff*w_ji_vel + learning_rate*dw_ji

        w_kj -= w_kj_vel
        w_ji -= w_ji_vel

        return w_kj, w_ji, w_kj_vel, w_ji_vel
    else:
        w_kj = w_kj - learning_rate * dw_kj
        w_ji = w_ji - learning_rate * dw_ji
        return w_kj, w_ji, w_kj_vel, w_ji_vel

X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = (X_train/127.5)-1 , (X_test/127.5)-1      #Converting pixel values from [0,255] to [-1,1]
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train[:,:], Y_train[:,:], 0.1)

# Hyperparameters
batch_size = 128
learning_rate = 0.5
num_batches = X_train.shape[0] // batch_size
should_check_gradient = False
check_step = num_batches // 10
max_epochs = 5
hidden_layer_units = 64

#Task3 parameters
shuffle_after_epoch = True          #3a
use_improved_sigmoid = True         #3b
smart_weight_initialization = True  #3c
use_momentum = True                 #3d
momentum_coeff = 0.9                #3d

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []

def train_loop():
    w_kj = weight_initialization(hidden_layer_units,Y_train.shape[1])
    w_ji = weight_initialization(X_train.shape[1],hidden_layer_units)

    w_kj_vel = np.zeros((Y_train.shape[1],hidden_layer_units))
    w_ji_vel = np.zeros((hidden_layer_units,X_train.shape[1]))


    for e in range(max_epochs): # Epochs

        if (shuffle_after_epoch):
            print("Randomizing training data...")
            shuffle_data(X_train,Y_train)

        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            hidden_layer = forward_hidden(X_batch,w_ji)

            w_kj, w_ji, w_kj_vel, w_ji_vel = gradient_descent(hidden_layer, Y_batch, X_batch, w_kj, w_ji,  learning_rate, should_check_gradient, w_kj_vel, w_ji_vel)

            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w_ji, w_kj))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w_ji, w_kj))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w_ji, w_kj))


                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w_ji, w_kj))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w_ji, w_kj))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w_ji, w_kj))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return w_ji, w_kj
    return w_ji, w_kj

w_ji, w_kj = train_loop()

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.legend()
plt.show()

plt.clf()