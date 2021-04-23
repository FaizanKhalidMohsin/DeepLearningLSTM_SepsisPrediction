#### Packages ######################
import argparse 
import logging

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
import sklearn as skm
from sklearn.utils import resample

import nni

#### Logging #######################
LOG = logging.getLogger('LSTM_experiment')
tf.keras.backend.set_image_data_format('channels_last')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']


#### HYPERPARAMETERS FOR DATA #######################
train_path = "../../Data/train_sample_and_hold_NA_constant.csv"
val_path = "../../Data/val_sample_and_hold_NA_constant.csv"
epochs = 50
es_patience = 10

# Functions
def drop_missing(train, val, threshold):
    # determine features based on training set
    percent_missing = train.apply(lambda x: sum(x == -1) / len(x))
    features_to_drop = percent_missing[percent_missing >= threshold].index.values
    # tell us what you're dropping
    LOG.debug("Dropping Features:\n", features_to_drop)
    # drop features in training and val
    train_dropped = train.drop(features_to_drop, axis=1, inplace=False)
    val_dropped = val.drop(features_to_drop, axis=1, inplace=False)

    return (train_dropped, val_dropped)


# Upsampling
def upsampling(data, ratio=1, set_seed=123):
    if ratio == 0:
        upsampled_data = data

    else:
        # separate minority and majority classes
        not_sepsis = data[data.SepsisLabel == 0]
        sepsis = data[data.SepsisLabel == 1]

        # get unique patients
        len_non_sepsis_ptx = len(np.unique(not_sepsis.index.values))
        sepsis_ptx = np.unique(sepsis.index.values)

        # upsample sepsis patients to desired ratio vs # of not_sepsis patients
        sepsis_ptx_upsampled = resample(sepsis_ptx,
                                        replace=True,  # sample with replacement
                                        n_samples=int(len_non_sepsis_ptx * replacement_ratio),
                                        # custom proportion of not_sepsis
                                        random_state=set_seed)  # reproducible results

        # new dataframe constructed from resampled patient list
        sepsis_upsampled = pd.concat(
            [sepsis.groupby('filename').get_group(patient) for patient in sepsis_ptx_upsampled])

        # combine majority and upsampled minority
        upsampled_data = pd.concat([not_sepsis, sepsis_upsampled])

    return upsampled_data


# Convert pd dataframe to LSTM format, adapted from MachineLearningMastery
def series_to_supervised(data, n_in=12, n_out=1, dropnan=True):
    values = data.values
    n_vars = 1 if values.ndim == 1 else data.shape[1]
    df = pd.DataFrame(values)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Convert pd dataframe into LSTM input
def LSTM_format(data, num_of_past_hours):

    # Note that this assumes that SepsisLabel is the last column of the dataframe
    data_X, data_y = data.iloc[:, :-1], data.iloc[:, -1]

	# Prepare for LSTM format
    data_X = data_X.groupby('filename').apply(series_to_supervised, n_in=num_of_past_hours).values
    data_y = data_y.groupby('filename').apply(series_to_supervised, n_in=num_of_past_hours).iloc[:, -1].values

    # reshape into 3d LSTM input (obs, timesteps, features)
    data_X = data_X.reshape((data_X.shape[0], num_of_past_hours + 1, int(data_X.shape[1] / (num_of_past_hours + 1))))

    return data_X, data_y


def confusion_matrix(model, data_X, data_y):
    pred_y = model.predict_classes(data_X)
    cmat = tf.math.confusion_matrix(
        data_y,
        pred_y,
    )
    cmat = pd.DataFrame(cmat.numpy())
    cmat.rename(columns={0: 'Predicted Sepsis: No', 1: 'Predicted Sepsis: Yes'},
                            index={0: 'Actual Sepsis: No', 1: 'Actual Sepsis: Yes'},
                            inplace=True)
	
    # components of confusion matrix
    TN = cmat.iloc[0,0]
    TP = cmat.iloc[1,1]
    FN = cmat.iloc[1,0]
    FP = cmat.iloc[0,1]

    # Other metrics
    SP = TN / (TN + FP) 
    SN = TP / (TP + FN) 
    PPV = TP / (TP + FP) 

    Acc = (TN + TP) / (TN + TP + FN + FP)
    weightAcc = (TN*0 + TP*1) / (TN*0 + TP*1 + FN*2 + FP*0.05)
    F1 = 2 * (PPV * SN) / (PPV + SN)

    LOG.debug(cmat)
    LOG.debug(
        '\n\nSpecificity:\t' + '{:.3}'.format(SP),
        '\nSensitivity:\t' + '{:.3}'.format(SN),
        '\nPrecision:\t' + '{:.3}'.format(PPV)
    )
    LOG.debug(
        '\nF1 Score:\t' + '{:.3}'.format(F1),
        '\nAccuracy:\t' + '{:.3}'.format(Acc),
        '\nSepsis-Weighted Accuracy:\t' + '{:.3}'.format(weightAcc)
    )

    return weightAcc, SN, F1

#class SendMetrics(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#	    LOG.debug(logs)
#	    nni.report_intermediate_result(logs["val_loss"])


def load_data(train_path, val_path, missing_threshold, replacement_ratio, num_of_past_hours):

	# Data Import
	train_data = pd.read_csv(train_path)
	train_data = train_data[train_data.filename.isin(train_data.filename.unique()[0:1001])].set_index('filename').drop(['Unit1','Unit2', 'GotSepsis', 'Hospital'], axis=1)

	val_data = pd.read_csv(val_path)
	val_data = val_data[val_data.filename.isin(val_data.filename.unique()[0:501])].set_index('filename').drop(['Unit1','Unit2', 'GotSepsis', 'Hospital'], axis=1)

	# Drop features missing above threshold
	train_dropped, val_dropped = drop_missing(train_data, val_data, threshold = missing_threshold)

	# Upsample septic cases to desired ratio vs non-septic
	upsampled_train = upsampling(train_dropped, ratio = replacement_ratio, set_seed = 27)

	# Convert to LSTM format
	train_X, train_y = LSTM_format(upsampled_train, num_of_past_hours)
	val_X, val_y = LSTM_format(val_dropped, num_of_past_hours)

	return train_X, train_y, val_X, val_y


# MODEL ARCHITECTURE
def LSTM_Model_Architecture(train_X, lstm_layers, dense_layers, lstm_layer_size, dense_layer_size, dropout):

    # Input
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.LSTM(lstm_layer_size, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences= True))
    lstm_model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    lstm_model.add(tf.keras.layers.Dropout(dropout))

    # Hidden Layers
    if lstm_layers <= 2:
        for num_lstm_layers in range(lstm_layers-1):
            lstm_model.add(tf.keras.layers.LSTM(lstm_layer_size))
            lstm_model.add(tf.keras.layers.Dropout(dropout))
    else:
        for num_lstm_layers in range(lstm_layers-2):
            lstm_model.add(tf.keras.layers.LSTM(lstm_layer_size, return_sequences= True))
            lstm_model.add(tf.keras.layers.Dropout(dropout))
        lstm_model.add(tf.keras.layers.LSTM(lstm_layer_size))
        lstm_model.add(tf.keras.layers.Dropout(dropout))

    for num_dense_layers in range(dense_layers):
        lstm_model.add(tf.keras.layers.Dense(dense_layer_size))
        lstm_model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        lstm_model.add(tf.keras.layers.Dropout(dropout))

    # Output
    lstm_model.add(tf.keras.layers.Dense(1, activation= 'sigmoid'))

    return lstm_model

def generate_default_params():
    return {
	    "missing_threshold":0.7,
		"num_of_past_hours":11,
		"replacement_ratio":1,
		"lstm_layers":2,
		"dense_layers":1,
		"lstm_layer_size":128,
		"dense_layer_size":64,
		"sepsis_weight":50,
		"dropout":0.5,
		"momentum":0,
		"decay":0,
		"batch_size":64,
		"optimizer":"RMSprop",
		"lr":0.001
		}

if __name__ == '__main__': 
	try:
	    # NNI stuff
		RECEIVED_PARAMS = nni.get_next_parameter()
		LOG.debug(RECEIVED_PARAMS)
		hyper_params = generate_default_params()
		hyper_params.update(RECEIVED_PARAMS)

        ### HYPERPARAMETERS FROM SEARCH SPACE
		missing_threshold = hyper_params['missing_threshold']
		num_of_past_hours = hyper_params['num_of_past_hours']
		replacement_ratio = hyper_params['replacement_ratio']
		lr = hyper_params['lr'] 
		decay = hyper_params['decay']
		batch_size = hyper_params['batch_size']
		momentum = hyper_params['momentum']
		sepsis_weight = hyper_params['sepsis_weight']
		dropout = hyper_params['dropout']
		optimizer= hyper_params['optimizer']
		lstm_layers = hyper_params['lstm_layers']
		dense_layers = hyper_params['dense_layers']
		lstm_layer_size = hyper_params['lstm_layer_size']
		dense_layer_size = hyper_params['dense_layer_size']

        # load data
		train_X, train_y, val_X, val_y = load_data(train_path, val_path, missing_threshold, replacement_ratio, num_of_past_hours)
	
	    # Train
		model = LSTM_Model_Architecture(train_X, 
		        lstm_layers = lstm_layers,
		        dense_layers = dense_layers,
				lstm_layer_size = lstm_layer_size,
				dense_layer_size = dense_layer_size, dropout = dropout)

		optimizer_choices = {"RMSprop":tf.keras.optimizers.RMSprop(learning_rate=lr),
                "Adam":tf.keras.optimizers.Adam(learning_rate=lr, decay=decay),
                "Adadelta":tf.keras.optimizers.Adadelta(learning_rate=lr, decay=decay),
                "SGD":tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, decay=decay)}

		model.compile(optimizer= optimizer_choices[optimizer],
		        loss= 'binary_crossentropy',
                metrics=['accuracy'])

		ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=es_patience)

		model.fit(train_X, train_y,
		        epochs= epochs,
				batch_size=batch_size,
				validation_data=(val_X, val_y),
				verbose=2,
				class_weight={0:1, 1:sepsis_weight},
				callbacks=[ES, tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR)])#, SendMetrics()])

		# get model accuracy
		weighted_acc, sn, f1 = confusion_matrix(model, val_X, val_y)
		LOG.debug('Final weighted accuracy is: %d', weighted_acc)
		nni.report_final_result({'default': weighted_acc, "sensitivity": sn, "f1_score": f1})

	except Exception as e:
	    LOG.exception(e)
	    raise
