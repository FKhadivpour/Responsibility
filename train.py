import numpy as np
from tensorflow.keras import datasets, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten  
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.regularizers import l2
import time
import pickle

def get_train_config():
    """Returns a default config file"""
    config = {
        'seed': 42,
        'dataset': 'mnist',
        'num_classes': 10,
        "batch_size": 1,
        "maxepoches": 3,
        "min_lr": 0.0001,
        "max_lr": 0.001,
        "img_shape": [28,28,1]}
    return config

def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0],28, 28, 1) 
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')   

    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(y_train, 10) 
    Y_test = to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test


def create_model(img_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=img_shape))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model


def mini_batches(X, Y, mini_batch_size=64):
    # Generate batches
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 2: Partition. Minus the end case.
    num_complete_minibatches = int(np.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def lr_scheduler(epoch, maxepoch, max_lr=0.01, min_lr=0.005):
      learning_rate = max_lr + (epoch * (min_lr - max_lr) / maxepoch)
      return learning_rate

def train(config, X_train, Y_train, X_test, Y_test): 

    total_start = time.time() 
    batch_size = config["batch_size"]
    maxepoch = config["maxepoches"]
    min_lr = config["min_lr"]
    max_lr = config["max_lr"]
    img_shape = config["img_shape"]
    num_classes = config["num_classes"]

    num_samples = (X_train.shape)[0]
    num_minibatches = int(np.ceil(num_samples / batch_size))

    total_delta_weights = [0] * num_samples
    print("Number of MiniBatches:", num_minibatches)

    model = create_model(img_shape=img_shape, num_classes=num_classes)

    opt = optimizers.SGD(lr=max_lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file="/content/drive/MyDrive/cifar10alexnet/New_Dataset/" + model_type + ".png", show_shapes=True, show_layer_names=True)
  
    for epoch in range(maxepoch):
        start = time.time()
        minibatches = mini_batches(X_train, Y_train, batch_size)

        learning_rate = lr_scheduler(epoch=epoch, maxepoch=maxepoch, max_lr=max_lr, min_lr=min_lr)

        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        print(f"---------------------- Epoch: {epoch+1} Learning Rate: {learning_rate} ----------------------")

        count = 0

        for minibatch in minibatches:
        
            (train_sample_x, train_sample_y) = minibatch

            pre_weights = np.array(model.layers[-1].get_weights()[0])

            model.fit(train_sample_x, train_sample_y,
                batch_size=batch_size,
                shuffle=False, 
                epochs=1,
                verbose=0)
        
            post_weights = np.array(model.layers[-1].get_weights()[0])


            delta_weights = np.subtract(post_weights, pre_weights)

            for i in range(len(train_sample_x)):
                total_delta_weights[i + count] = np.add(total_delta_weights[i + count], delta_weights)

            count += len(train_sample_x)
    
        print("Train Scores:")
        _, _ = model.evaluate(x=X_train, y=Y_train, verbose=1)
        print("Test Scores:")
        _, _ = model.evaluate(x=X_test, y=Y_test, verbose=1) 
        
        end = time.time()
        print("Epoch time = ", start - end)

    print("Total training time = ", time.time() - total_start)
    return model, total_delta_weights


def save_model(model):
    PATH = 'keras_model'
    model.save(PATH)


def save_delta_weights(total_delta_weights):
    pickle.dump(total_delta_weights, open("total_delta_weights.pickle", "wb"))


if __name__ == "__main__":
    config = get_train_config()
    X_train, Y_train, X_test, Y_test = load_data()
    model, total_delta_weights = train(config, X_train, Y_train, X_test, Y_test)
    save_model(model)
    save_delta_weights(total_delta_weights)