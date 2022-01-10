
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path
import random
from collections import Counter



def get_config():
    """Returns a default config file"""
    config = {
        'data_path': 'data',
        'critic_data_path': 'critic_data',
        'dataset': 'cifar10',
        'actor_model_path': 'keras_model_',
        'critic_model_path': 'critic_model_',
        'outdir': 'outdir',
        'seed': 42,
        'num_classes': 10,
        "img_shape": [32,32,3]}
    return config

def load_data(config):
    """Load Data and preprocess it"""
    X_train = pickle.load(open(config['data_path'] + "/X_train.pickle", "rb"))
    Y_train = pickle.load(open(config['data_path'] + "/Y_train.pickle", "rb"))
    X_test = pickle.load(open(config['data_path'] + "/X_test.pickle", "rb"))
    Y_test = pickle.load(open(config['data_path'] + "/Y_test.pickle", "rb"))
    X_val = pickle.load(open(config['data_path'] + "/X_val.pickle", "rb"))
    Y_val = pickle.load(open(config['data_path'] + "/Y_val.pickle", "rb"))

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def load_actor_model(config):
    """Load pretrained Keras model from the saved path"""
    PATH = config['actor_model_path'] + config['dataset']
    model = load_model(PATH)
    return model

def split_ids(X, Y, model):
    pred = np.argmax(model.predict(X), axis=1)
    y_true = np.argmax(Y, axis = 1)
    correct_ids = [i for i, v in enumerate(pred) if v == y_true[i]]
    incorrect_ids = [i for i, v in enumerate(pred) if v != y_true[i]]
    print(len(correct_ids), len(incorrect_ids))
    correct_ids_ = random.sample(correct_ids, len(incorrect_ids))
    return correct_ids_, incorrect_ids

def load_delta_weights(config):
    array_path = config['dataset'] + "_total_delta_weights.pickle"
    total_delta_weights = pickle.load(open(array_path, "rb"))
    return total_delta_weights

def responsible_ids(config, total_delta_weights):
    """Finds IDs of the training samples that maximally altered each neuron
    Arguments:
        config: dict, contains the configuration from cli params
        total_delta_weights: numpy array, Changes that each training instance made to each neuron. 
    Returns:
        positive_responsible_IDs: Numpy array of IDs of the training samples that maximally altered each neuron"""
    array = np.array(total_delta_weights, dtype=np.float16)

    positive_responsible_IDs = np.argmax(array, axis=0)
    print(f"Most Responsible Positive Array ---- Shape: {positive_responsible_IDs.shape}")
    counts_max = np.bincount(positive_responsible_IDs.flatten())
    print("Most Frequent Index:", counts_max.argsort()[-10:][::-1])

    negative_responsible_IDs = np.argmax(array, axis=0)

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    positive_responsible_IDs_path = outdir.joinpath(f"{config['dataset']}_positive_responsible_IDs.pickle")
    negative_responsible_IDs_path = outdir.joinpath(f"{config['dataset']}_negative_responsible_IDs.pickle")

    pickle.dump(positive_responsible_IDs, open(positive_responsible_IDs_path, "wb"))
    pickle.dump(negative_responsible_IDs, open(negative_responsible_IDs_path, "wb"))

    return positive_responsible_IDs, negative_responsible_IDs

def find_responsible_sample_wx_method(test_sample, model, positive_responsible_ids, training_samples):
    """Finds the most responsible training instance for a single test dataset image.
    Gets a test sample, the model and array of IDs of the responsible samples. 
    Calcualtes the `wx` values on the fly and discards them afterwards.
    Returns a training instance that is the most responsible for prediction of the model.
    Arguments:
        test_sample: A testing instance. 
        model: pretrained Keras model.
        training_samples: All training instances.
        positive_responsible_ids: Numpy array of IDs of the training samples that maximally altered each neuron.
    Returns:
        most_responsible_sample: a training instance that is the most responsible for prediction of the model"""
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functor = K.function([inp], outputs)  # evaluation function
    layer_outs = functor([np.reshape(test_sample, (1,32,32,3))]) ### TODO Should change the shape
    last_layer_output = layer_outs[-1]
    last_layer_input = layer_outs[-2].reshape((1024,)) ### TODO Should change the shape

    index_of_largest_value = np.argmax(last_layer_output)
    last_layer_weights = np.array(model.layers[-1].get_weights()[0])
    list_of_weights_led_to_largest_value = last_layer_weights[:, index_of_largest_value]

    w = []
    x = []
    wx = []
    for i in range(len(list_of_weights_led_to_largest_value)):
        w.append(list_of_weights_led_to_largest_value[i])
        x.append(last_layer_input[i])
        wx.append(list_of_weights_led_to_largest_value[i] * last_layer_input[i])

    most_responsible_sample = training_samples[positive_responsible_ids[np.argmax(wx), index_of_largest_value]]

    return most_responsible_sample

def generate_resp_tuples(config, X, correct_ids, incorrect_ids, train_samples, actor_model):
    tuples = []
    total_delta_weights = load_delta_weights(config)
    positive_responsible_IDs, _ = responsible_ids(config, total_delta_weights)
    for id in correct_ids:
        resp_sample = find_responsible_sample_wx_method(X[id], actor_model, positive_responsible_IDs, train_samples)
        tuples.append((X[id], resp_sample, 1.0))
    for id in incorrect_ids:
        resp_sample = find_responsible_sample_wx_method(X[id], actor_model, positive_responsible_IDs, train_samples)
        tuples.append((X[id], resp_sample, 0.0))
    np.random.seed(config['seed'])
    np.random.shuffle(tuples)
    print(f"{len(tuples)} responsible tuples!")
    return tuples
    

def euc(flatten_img1, flatten_img2):

    RH1 = Counter(flatten_img1)
    RH2 = Counter(flatten_img2)
    H1 = []
    for i in range(128):
        if i in RH1.keys():
            H1.append(RH1[i])
        else:
            H1.append(0)
    H2 = []
    for i in range(128):
        if i in RH2.keys():
            H2.append(RH2[i])
        else:
            H2.append(0)
    distance =0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)

def find_nearest_neighbor(sample, x_train, ids):
    min_dist = np.inf
    for i in ids:
        img = x_train[i]
        dist = euc(list(sample.flatten()), list(img.flatten()))
        if dist < min_dist:
            min_dist = dist
            most_sim_img = img
    return most_sim_img

def find_ids_of_each_class(num_classes, y):
    ids = {}
    for c in range(num_classes):
        ids[str(c)] = []
    for id in range(len(y)):
        ids[str(np.argmax(y[id]))].append(id)
    return ids
        

def generate_nn_tuples(config, X, correct_ids, incorrect_ids, x_train, y_train, actor_model):
    tuples = []
    class_ids = find_ids_of_each_class(config['num_classes'], y_train)
    n = 0
    for id in correct_ids:
        label = str(np.argmax(actor_model.predict(x_train[id: id+1])))
        ids = class_ids[label]
        nn_sample = find_nearest_neighbor(X[id], x_train, ids)
        tuples.append((X[id], nn_sample, 1.0))
        if (n+1) % 100 == 0:
            print('{} out of {} correct samples.'.format(n+1, len(correct_ids)))
        n += 1
    n = 0
    for id in incorrect_ids:
        label = str(np.argmax(actor_model.predict(x_train[id: id+1])))
        ids = class_ids[label]
        nn_sample = find_nearest_neighbor(X[id], x_train, ids)
        tuples.append((X[id], nn_sample, 0.0))
        if (n+1) % 100 == 0:
            print('{} out of {} incorrect samples.'.format(n+1, len(incorrect_ids)))
        n += 1
    np.random.seed(config['seed'])
    np.random.shuffle(tuples)
    print(f"{len(tuples)} nearest neighbor tuples!")
    return tuples


if __name__ == "__main__":
    config = get_config()
    X_train, Y_train, X_test, Y_test, X_val, Y_val = load_data(config)
    actor_model = load_actor_model(config)
    actor_model.evaluate(X_val, Y_val, verbose=1)
    actor_model.evaluate(X_test, Y_test, verbose=1)
    val_correct_ids, val_incorrect_ids = split_ids(X_val, Y_val, actor_model)
    test_correct_ids, test_incorrect_ids = split_ids(X_test, Y_test, actor_model)
    print(f"{len(val_correct_ids)}/{len(val_incorrect_ids)} correct/incorrect validation ids for training!")
    print(f"{len(test_correct_ids)}/{len(test_incorrect_ids)} correct/incorrect test ids for unseen testing!")
    print("Let's generate dataset:")
    resp_tuples_train = generate_resp_tuples(config, X_val, val_correct_ids, val_incorrect_ids, X_train, actor_model)
    resp_tuples_test = generate_resp_tuples(config, X_test, test_correct_ids, test_incorrect_ids, X_train, actor_model)
    pickle.dump(resp_tuples_train, open(config['critic_data_path'] + "/resp_tuples_train.pickle", "wb"), protocol=4)
    pickle.dump(resp_tuples_test, open(config['critic_data_path'] + "/resp_tuples_test.pickle", "wb"), protocol=4)
    print("Responsibility dataset generated!")
    nn_tuples_train = generate_nn_tuples(config, X_val, val_correct_ids, val_incorrect_ids, X_train, Y_train, actor_model)   
    nn_tuples_test = generate_nn_tuples(config, X_test, test_correct_ids, test_incorrect_ids, X_train, Y_train, actor_model)
    pickle.dump(nn_tuples_train, open(config['critic_data_path'] + "/nn_tuples_train.pickle", "wb"), protocol=4)
    pickle.dump(nn_tuples_test, open(config['critic_data_path'] + "/nn_tuples_test.pickle", "wb"), protocol=4)
    print("Nearest Neighbor dataset generated!")

