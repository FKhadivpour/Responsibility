from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import pickle
import json

def get_config():
    """Returns a default config file"""
    config = {
        'json_path': 'outdir/responsibility_results_meta-10.json',
        'pickle_path': './outdir/responsibles_results_10.pickle',
        'test_sample_num': 10,
        'num_classes': 10,
        'num_images_to_plot':2, # should be < test_sample_num from calc_responsible_samples.py which is 10
        }
    return config

def load_data():
    """Load Data and preprocess it"""
    (X_train, _), (X_test, _) = datasets.mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0],28, 28, 1) 
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')   

    X_train /= 255
    X_test /= 255

    return X_train, X_test

def load_json(config):
    """lodds a json file
    Arguments:
        config: dict, contains the configuration.
    """
    with open(config['json_path']) as f:
        json_file = json.load(f)

    return json_file

def load_pickle(config):
    """lodds a pickle file
    Arguments:
        config: dict, contains the configuration.
    """
    with open(config['pickle_path'], 'rb') as f:
        pickle_file = pickle.load(f)

    return pickle_file

def plot_sample(sample, most_resp_sample):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].imshow(sample.reshape((28,28)), cmap='gray')
    ax[1].imshow(most_resp_sample.reshape((28,28)), cmap='gray')
    plt.show()


def plot_responsibility(responsibles, config, X_test):
    keys = responsibles.keys()
    for i in range(config['num_classes']):
        for j in range(config['num_images_to_plot']):
            k = list(keys)[int(str(i)+str(j))]
            num_in_dataset = responsibles[k]['num_in_dataset']
            print(f"{num_in_dataset}) test sample number {k}")
            sample = X_test[int(k)]
            most_resp_sample = responsibles[k]['most_responsible_sample']
            plot_sample(sample, most_resp_sample)
        
        

if __name__ == "__main__":
    config = get_config()
    responsibles = load_pickle(config)
    X_train, X_test = load_data()
    plot_responsibility(responsibles, config, X_test)



