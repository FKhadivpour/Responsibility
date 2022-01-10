import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime as dt
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten  
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def get_config():
    """Returns a default config file"""
    config = {
        'critic_data_path': 'critic_data',
        'actor_model_path': 'keras_model_',
        'critic_path': 'critic_results',
        'critic_model_path': 'critic_model_',
        'seed': 42,
        'num_classes': 2,
        "img_shape": [32,32,3],
        "concat_img_shape": [32,32,6],
        "batch_size": 16,
        "min_lr": 0.00001,
        "max_lr": 0.0001,
        "maxepoch": 40
        }
    return config

def save_model(model, config, model_type):
    path = config['critic_model_path'] + model_type
    model.save(path)


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
    for key in json_obj.keys():
        print(key, json_obj[key])

def load_data(config, model_type):
    if model_type == 'original':
        train_tuples = pickle.load(open(config['critic_data_path'] + "/resp_tuples_train.pickle", "rb"))
        unseen_tuples = pickle.load(open(config['critic_data_path'] + "/resp_tuples_test.pickle", "rb"))
        x = []
        y = []
        for tpl in train_tuples:
            x.append(tpl[0])
            if tpl[2] == 0:
                y.append([1.0,0.0])
            elif tpl[2] == 1:
                y.append([0.0,1.0])

        for unseen_tpl in unseen_tuples:
            x.append(unseen_tpl[0])
            if unseen_tpl[2] == 0:
                y.append([1.0,0.0])
            elif unseen_tpl[2] == 1:
                y.append([0.0,1.0])
       
        x = np.array(x)
        y = np.array(y)

        x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, stratify=y, random_state=config['seed'])


        print('Training Data:')
        print("Input Shapes:", x_tr.shape)
        print("Output shape:", y_tr.shape)

        print('Unseen Data:')
        print("Input Shapes:", x_ts.shape)
        print("Output shape:", y_ts.shape)
    
    elif model_type == 'responsibility':
        train_tuples = pickle.load(open(config['critic_data_path'] + "/resp_tuples_train.pickle", "rb"))
        unseen_tuples = pickle.load(open(config['critic_data_path'] + "/resp_tuples_test.pickle", "rb"))
        x = []
        y = []
        for tpl in train_tuples:
            x.append(np.concatenate((tpl[0], tpl[1]), axis=2))
            if tpl[2] == 0:
                y.append([1.0,0.0])
            elif tpl[2] == 1:
                y.append([0.0,1.0])

        for unseen_tpl in unseen_tuples:
            x.append(np.concatenate((tpl[0], tpl[1]), axis=2))
            if unseen_tpl[2] == 0:
                y.append([1.0,0.0])
            elif unseen_tpl[2] == 1:
                y.append([0.0,1.0])
       
        x = np.array(x)
        y = np.array(y)

        x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, stratify=y, random_state=config['seed'])


        print('Training Data:')
        print("Input Shapes:", x_tr.shape)
        print("Output shape:", y_tr.shape)

        print('Unseen Data:')
        print("Input Shapes:", x_ts.shape)
        print("Output shape:", y_ts.shape)
    
    elif model_type == 'nearest_neighbor':
        train_tuples = pickle.load(open(config['critic_data_path'] + "/nn_tuples_train.pickle", "rb"))
        unseen_tuples = pickle.load(open(config['critic_data_path'] + "/nn_tuples_test.pickle", "rb"))
        x = []
        y = []
        for tpl in train_tuples:
            x.append(np.concatenate((tpl[0], tpl[1]), axis=2))
            if tpl[2] == 0:
                y.append([1.0,0.0])
            elif tpl[2] == 1:
                y.append([0.0,1.0])

        for unseen_tpl in unseen_tuples:
            x.append(np.concatenate((tpl[0], tpl[1]), axis=2))
            if unseen_tpl[2] == 0:
                y.append([1.0,0.0])
            elif unseen_tpl[2] == 1:
                y.append([0.0,1.0])
       
        x = np.array(x)
        y = np.array(y)

        x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, stratify=y, random_state=config['seed'])


        print('Training Data:')
        print("Input Shapes:", x_tr.shape)
        print("Output shape:", y_tr.shape)

        print('Unseen Data:')
        print("Input Shapes:", x_ts.shape)
        print("Output shape:", y_ts.shape)
    return x_tr, y_tr, x_ts, y_ts

def build_model(config, model_type):
    if model_type == 'original':
        input_shape = config['img_shape']
    else: 
        input_shape = config['concat_img_shape']
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(config['num_classes'], activation='softmax'))
    return model

def lr_scheduler(maxepoch, max_lr, min_lr):
      learning_rate = max_lr + ((min_lr - max_lr) / (maxepoch))
      print("Learning Rate:", learning_rate)
      return learning_rate

def train(config, X_train, Y_train, model_type):
    model = build_model(config, model_type)
    def lr_scheduler(epoch):
      if epoch <= 0.6 * config['maxepoch']:
        learning_rate = config['max_lr']
      else:
        learning_rate = config['max_lr'] + ((epoch+1 - (0.6 * config['maxepoch'])) * (config['min_lr'] - config['max_lr']) / (0.4 * config['maxepoch']) )
      print("Epoch:", epoch, "Learning Rate:", learning_rate)
      return learning_rate
    reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=3)
    opt = optimizers.Adam(learning_rate=config['max_lr'])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, Y_train,
                    batch_size=config['batch_size'],
                    shuffle=True, 
                    epochs=config['maxepoch'],
                    validation_split=0.1, 
                    callbacks=[reduce_lr, earlystop],
                    verbose=1)
    return model, history

def test(model, hist, X_test, Y_test, plot=False):
    accuracy, loss = model.evaluate(X_test, Y_test)
    y_hat = model.predict(X_test)

    y_pred = np.argmax(y_hat, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if plot == True:
        print(hist.history.keys())
        # summarize history for accuracy
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    return accuracy, loss, precision, recall, f1
    

if __name__ == "__main__":
    config = get_config()

    scores = {'accuracy':{}, 'loss':{}, 'precision':{}, 'recall':{}, 'f1':{}}

    for model_type in ['original', 'responsibility']:
        print(f"------------- Model type: {model_type} ------------")
        X_train, Y_train, X_test, Y_test = load_data(config, model_type)
        print(f'{model_type} data loaded!')
        model, history = train(config, X_train, Y_train, model_type)
        save_model(model, config, model_type)
        print(f'Training with {model_type} data done!')
        scores['accuracy'][model_type], scores['loss'][model_type],  scores['precision'][model_type], scores['recall'][model_type], scores['f1'][model_type] = test(model, history, X_test, Y_test, plot=False)

    outdir = Path(config['critic_path'])
    outdir.mkdir(exist_ok=True, parents=True)
    scores_fn = "critic_scores.json"
    scores_path = outdir.joinpath(scores_fn)
    save_json(scores, scores_path)

