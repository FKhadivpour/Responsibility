
import numpy as np
from tensorflow.keras import datasets
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import time
import pickle
import json
import copy
from pathlib import Path
from datetime import datetime as dt


def get_test_config():
    """Returns a default config file"""
    config = {
        'num_classes': 10,
        'model_path': 'keras_model',
        'outdir': 'outdir',
        'test_sample_num': 10,
        "img_shape": [28,28,1]}
    return config

def load_data():
    """Load Data and preprocess it"""
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

def load_my_model(config):
    """Load pretrained Keras model from the saved path"""
    model = load_model(config['model_path'])
    return model

def load_delta_weights():

    total_delta_weights = pickle.load(open("total_delta_weights.pickle", "rb"))
    return total_delta_weights

def save_json(json_obj, json_path, append_if_exists=False,
              overwrite_if_exists=False, unique_fn_if_exists=True):
    """Saves a json file
    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}'\
                                               f'{str(json_path.suffix)}'

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=2)

def get_dataset_sample_ids_per_class(class_id, num_samples, X_test, Y_test):
    """Gets the first num_samples from class class_id. 
    Returns a list with the indicies which can be passed to X_train to retreive the actual data.
    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        X_test, Y_test: test dataset.
    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    sample_list = []
    img_count = 0
    for i in range(len(X_test)):
        t = np.argmax(Y_test[i])
        if class_id == t:
            sample_list.append(i)
            img_count += 1
            if img_count == num_samples:
                break

    return sample_list

def get_dataset_sample_ids(test_sample_num, X_test, Y_test, num_classes):
    """Gets the first num_sample indices of all classes. 
    Returns a list and a dict containing the indicies.
    Arguments:
        num_samples: int, number of samples of each class to return
        X_train, Y_train: test dataset.
        num_classes: int, number of classes contained in the dataset
    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    sample_list = []
    for i in range(num_classes):
        sample_dict[str(i)] = get_dataset_sample_ids_per_class(i,test_sample_num, X_test, Y_test)
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list):len(sample_list)] = sample_dict[str(i)]
    return sample_dict, sample_list

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

    positive_responsible_IDs_path = outdir.joinpath(f"positive_responsible_IDs.pickle")
    negative_responsible_IDs_path = outdir.joinpath(f"negative_responsible_IDs.pickle")

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
    layer_outs = functor([np.reshape(test_sample, (1,28,28,1))]) ### TODO Should change the shape
    last_layer_output = layer_outs[-1]
    last_layer_input = layer_outs[-2].reshape((84,)) ### TODO Should change the shape

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

def find_responsibles(config, model, X_train, X_test, Y_test, positive_responsible_ids):
    """Calculates the most responsible training instances, one test point at a time. 
    Arguments:
        config: dict, contains the configuration from cli params
        model: pretrained Keras model.
        X_train: Training input images. 
        X_test, Y_testL: Test dataset.
        positive_responsible_ids: Numpy array of IDs of the training samples that maximally altered each neuron.
        """

    rersponsible_meta = copy.deepcopy(config)
    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    
    test_sample_num = config['test_sample_num']
    num_classes = config['num_classes']
    _, sample_list = get_dataset_sample_ids(test_sample_num, X_test, Y_test, num_classes)
    test_dataset_iter_len = len(sample_list)

    # Set up logging and save the metadata conf file
    rersponsible_meta['test_sample_index_list'] = sample_list
    rersponsible_meta_fn = f"responsibility_results_meta-{test_sample_num}.json"
    rersponsible_meta_path = outdir.joinpath(rersponsible_meta_fn)
    save_json(rersponsible_meta, rersponsible_meta_path)

    responsibles = {}
    # Main loop for calculating the most responsible instnces. One test sample per iteration.
    for j in range(test_dataset_iter_len):
        i = sample_list[j]

        start_time = time.time()
        most_responsible_sample = find_responsible_sample_wx_method(X_test[i], model, 
                                                                    positive_responsible_ids,
                                                                    X_train)
        end_time = time.time()

        responsibles[str(i)] = {}
        label = np.argmax(Y_test[i])
        responsibles[str(i)]['label'] = label
        responsibles[str(i)]['num_in_dataset'] = j
        responsibles[str(i)]['time_calc_influence_s'] = end_time - start_time
        responsibles[str(i)]['most_responsible_sample'] = most_responsible_sample

    responsibles_path = outdir.joinpath(f"responsibles_results_{test_sample_num}.pickle")
    pickle.dump(responsibles, open(responsibles_path, "wb"))

    """### TODO ###
    save_json(responsibles, responsibles_path)
    """

if __name__ == "__main__":
    config = get_test_config()
    model = load_my_model(config)
    X_train, Y_train, X_test, Y_test = load_data()
    total_delta_weights = load_delta_weights()
    positive_responsible_IDs, _ = responsible_ids(config, total_delta_weights)
    find_responsibles(config, model, X_train, X_test, Y_test, positive_responsible_IDs)