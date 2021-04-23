#### Packages ######################
import argparse
import logging

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
from sklearn.utils import resample
from sklearn.metrics import average_precision_score, roc_auc_score

import nni

#### Logging #######################
LOG = logging.getLogger('LSTM_experiment')
tf.keras.backend.set_image_data_format('channels_last')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']

#### HYPERPARAMETERS FOR DATA #######################
data_dir = "~/Data/"
epochs = 100
es_patience = 15

# Functions
def drop_missing(train, val, threshold):
	# determine features based on training set
	percent_missing = train.apply(lambda x: sum(x == -1) / len(x))
	features_to_drop = percent_missing[percent_missing >= threshold].index.values
	# tell us what you're dropping
	print("Dropping Features:\n", features_to_drop)
	LOG.debug("Dropping Features:\n", features_to_drop)
	# drop features in training and val
	train_dropped = train.drop(features_to_drop, axis=1, inplace=False)
	val_dropped = val.drop(features_to_drop, axis=1, inplace=False)

	return train_dropped, val_dropped


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
										random_state=set_seed)	# reproducible results
		# new dataframe constructed from resampled patient list
		sepsis_upsampled = pd.concat(
			[sepsis.groupby('filename').get_group(patient) for patient in sepsis_ptx_upsampled])
		# combine majority and upsampled minority
		upsampled_data = pd.concat([not_sepsis, sepsis_upsampled])

	return upsampled_data


# Pad and convert into RNN-friendly format
def series_to_supervised(data, n_in, extra_padding, n_out=1, dropnan=True):
	# pad first
	if (n_in+n_out) > data.shape[0]:
		data = pd.DataFrame(data)
		rows_to_pad = int(n_in + n_out - data.shape[0] + data.shape[0]*extra_padding)
		padding = pd.DataFrame(-1, index=np.arange(rows_to_pad), columns=data.columns)
		padding.index = np.repeat(data.index.unique(), padding.shape[0])
		data = pd.concat([padding,data])
	# then run function
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
def LSTM_format(data, num_of_past_hours, extra_padding):
	# Note that this assumes that SepsisLabel is the last column of the dataframe
	data_X, data_y = data.iloc[:, :-1], data.iloc[:, -1]
	# Prepare for LSTM format
	data_X = data_X.groupby('filename').apply(series_to_supervised, n_in=num_of_past_hours, extra_padding=extra_padding).values
	data_y = data_y.groupby('filename').apply(series_to_supervised, n_in=num_of_past_hours, extra_padding=extra_padding).iloc[:, -1].values
	# reshape into 3d LSTM input (obs, timesteps, features)
	data_X = data_X.reshape((data_X.shape[0], num_of_past_hours + 1, int(data_X.shape[1] / (num_of_past_hours + 1))))

	return data_X, data_y


def get_samp_score(samp):
    samp = samp.fillna(0)
    n = len(samp)
    if np.sum(samp.SepsisLabel) > 0:
        t_sepsis = int(samp[samp.SepsisLabel == 1].ICULOS.min()) + 6
    else:
        t_sepsis = np.inf

    dt_early = -12
    dt_optimal = -6
    dt_late = 3
    m_1 = 1 / (dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = -1 / (3 - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = -2 / (dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u = 0
    for t in range(1, n):
        if t <= t_sepsis + dt_late:
            # TP
            if samp.iloc[t,:]['SepsisLabel'] == 1 and samp.iloc[t,:]['SepsisPred'] == 1:
                if t <= t_sepsis + dt_optimal:
                    u += max(m_1 * (t - t_sepsis) + b_1, -0.05)
                elif t <= t_sepsis + dt_late:
                    u += m_2 * (t - t_sepsis) + b_2
            # FN
            elif samp.iloc[t,:]['SepsisLabel'] == 1 and samp.iloc[t,:]['SepsisPred'] == 0:
                if t <= t_sepsis + dt_optimal:
                    u +=  0
                elif t <= t_sepsis + dt_late:
                    u +=  m_3 * (t - t_sepsis) + b_3
            # FP
            elif samp.iloc[t,:]['SepsisLabel'] == 0 and samp.iloc[t,:]['SepsisPred'] == 1:
                u += -0.05
            # TN
            elif samp.iloc[t,:]['SepsisLabel'] == 0 and samp.iloc[t,:]['SepsisPred'] == 0:
                u += 0

    return pd.Series({'score': u})


def evaluate_model(model, val_X, val_y, val_before_reshape, num_of_past_hours, extra_padding):
	# get pred
	prob_y = model.predict(val_X)
	pred_y = model.predict_classes(val_X)
	AP = average_precision_score(val_y, prob_y)
	AUROC = roc_auc_score(val_y, prob_y)
	### CONFUSION MATRIX ############
	cmat = tf.math.confusion_matrix(val_y, pred_y)
	cmat = pd.DataFrame(cmat.numpy())
	cmat.rename(columns={0: 'Predicted Sepsis: No', 1: 'Predicted Sepsis: Yes'},
                            index={0: 'Actual Sepsis: No', 1: 'Actual Sepsis: Yes'}, inplace=True)
    # components of confusion matrix
	TN = cmat.iloc[0,0]
	TP = cmat.iloc[1,1]
	FN = cmat.iloc[1,0]
	FP = cmat.iloc[0,1]
    # Other metrics
	SN = TP / (TP + FN) 
	PPV = TP / (TP + FP) 
	ACC = (TN + TP) / (TN + TP + FN + FP)
	F1 = 2 * (PPV * SN) / (PPV + SN)

	### UTILITY SCORES ############
	val_true = val_before_reshape.iloc[:,-2:]
	key = pd.DataFrame(val_true.groupby('filename').apply(series_to_supervised, n_in=num_of_past_hours, extra_padding=extra_padding).iloc[:,-2:])
	key.columns = ['ICULOS', 'SepsisLabel']
	key['SepsisPred'] = pred_y
	compare = val_true.reset_index().merge(key, how='left', on = ['filename', 'ICULOS', 'SepsisLabel']).set_index('filename')
	utility_score = pd.DataFrame(compare.groupby('filename').apply(get_samp_score)).score.sum()

    # number of septic patients
	n_septic = compare.groupby('filename').apply(lambda x: sum(x.SepsisLabel) > 1).sum()
	top_score = n_septic * (3 + 4.5)
	worst_score = n_septic * (-2 * 4.5)
	normalized_score = (utility_score - worst_score) / (top_score - worst_score)
	
	print(
		"\n\nUtility Score:\t" + '{:.3}'.format(utility_score) +
		"\nTop Score:\t" + '{:.3}'.format(top_score) + 
		"\nWorst Score:\t" + '{:.3}'.format(worst_score) + 
		"\nNormalized Score:\t" + '{:.3}'.format(normalized_score)
	)

	print(cmat)
	print(
		'\n\nNormalized Score:\t' + '{:.3}'.format(normalized_score) +
		'\nAccuracy:\t' + '{:.3}'.format(ACC) +
		'\nSensitivity:\t' + '{:.3}'.format(SN) +
		'\nPrecision:\t' + '{:.3}'.format(PPV) +
		'\nF1 Score:\t' + '{:.3}'.format(F1) +
		'\nAUPRC:\t' + '{:.3}'.format(AP) +
		'\nAUROC:\t' + '{:.3}'.format(AUROC)
	)

	return normalized_score, utility_score, ACC, SN, PPV, F1, AP, AUROC


def load_data(train_path, val_path, missing_threshold, replacement_ratio):
	# Data Import
	train_data = pd.read_csv(train_path)
	train_data = train_data.set_index('filename').drop(['Unit1','Unit2', 'GotSepsis', 'Hospital'], axis=1)
	val_data = pd.read_csv(val_path)
	val_data = val_data.set_index('filename').drop(['Unit1','Unit2', 'GotSepsis', 'Hospital'], axis=1)
	# Drop features missing above threshold
	train_dropped, val_dropped = drop_missing(train_data, val_data, threshold = missing_threshold)
	# Upsample septic cases to desired ratio vs non-septic
	upsampled_train = upsampling(train_dropped, ratio = replacement_ratio, set_seed = 27)

	return upsampled_train, val_dropped


# MODEL ARCHITECTURE
def Model_Architecture(train_X, rnn_type, rnn_layers, dense_layers, rnn_layer_size, dense_layer_size, rnn_dropout, dense_dropout):
	# Input
	model = tf.keras.Sequential()

	if rnn_type == "LSTM":
		model.add(tf.keras.layers.LSTM(rnn_layer_size, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences= True))
		model.add(tf.keras.layers.Dropout(rnn_dropout))
		# Hidden Layers
		if rnn_layers <= 2:
			for num_rnn_layers in range(rnn_layers-1):
				model.add(tf.keras.layers.LSTM(rnn_layer_size))
				model.add(tf.keras.layers.Dropout(rnn_dropout))
		else:
			for num_rnn_layers in range(rnn_layers-2):
				model.add(tf.keras.layers.LSTM(rnn_layer_size, return_sequences= True))
				model.add(tf.keras.layers.Dropout(rnn_dropout))
			model.add(tf.keras.layers.LSTM(rnn_layer_size))
			model.add(tf.keras.layers.Dropout(rnn_dropout))

	elif rnn_type == "GRU":
		model.add(tf.keras.layers.GRU(rnn_layer_size, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences= True))
		model.add(tf.keras.layers.Dropout(rnn_dropout))
		# Hidden Layers
		if rnn_layers <= 2:
			for num_rnn_layers in range(rnn_layers-1):
				model.add(tf.keras.layers.GRU(rnn_layer_size))
				model.add(tf.keras.layers.Dropout(rnn_dropout))
		else:
			for num_rnn_layers in range(rnn_layers-2):
				model.add(tf.keras.layers.GRU(rnn_layer_size, return_sequences= True))
				model.add(tf.keras.layers.Dropout(rnn_dropout))
			model.add(tf.keras.layers.GRU(rnn_layer_size))
			model.add(tf.keras.layers.Dropout(rnn_dropout))

	for num_dense_layers in range(dense_layers):
		model.add(tf.keras.layers.Dense(dense_layer_size))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
		model.add(tf.keras.layers.Dropout(dense_dropout))
	# Output
	model.add(tf.keras.layers.Dense(1, activation= 'sigmoid'))

	return model


def generate_default_params():
	return {
		"imputation":"sample_and_hold",
		"missing_threshold":0.7,
		"num_of_past_hours":11,
		"replacement_ratio":1,
		"extra_padding":0,
		"rnn_type":"LSTM",
		"rnn_layers":2,
		"dense_layers":1,
		"rnn_layer_size":128,
		"dense_layer_size":64,
		"sepsis_weight":50,
		"rnn_dropout":0.5,
		"dense_dropout":0.5,
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
		print(RECEIVED_PARAMS)
		LOG.debug(RECEIVED_PARAMS)
		hyper_params = generate_default_params()
		hyper_params.update(RECEIVED_PARAMS)

		### HYPERPARAMETERS FROM SEARCH SPACE
		imputation = hyper_params['imputation']
		missing_threshold = hyper_params['missing_threshold']
		num_of_past_hours = hyper_params['num_of_past_hours']
		replacement_ratio = hyper_params['replacement_ratio']
		extra_padding = hyper_params['extra_padding']
		rnn_type = hyper_params['rnn_type']
		rnn_layers = hyper_params['rnn_layers']
		dense_layers = hyper_params['dense_layers']
		rnn_layer_size = hyper_params['rnn_layer_size']
		dense_layer_size = hyper_params['dense_layer_size']
		sepsis_weight = hyper_params['sepsis_weight']
		rnn_dropout = hyper_params['rnn_dropout']
		dense_dropout = hyper_params['dense_dropout']
		optimizer= hyper_params['optimizer']
		decay = hyper_params['decay']
		momentum = hyper_params['momentum']
		lr = hyper_params['lr'] 
		batch_size = hyper_params['batch_size']

		# load data
		if imputation == "sample_and_hold":
			train_path = data_dir+"train_sample_and_hold_NA_constant.csv"	
			val_path = data_dir+"val_sample_and_hold_NA_constant.csv"		
		elif imputation == "ffill":
			train_path = data_dir+"train_cleaned_ffill_NA_constant.csv"
			val_path = data_dir+"val_cleaned_ffill_NA_constant.csv"

		upsampled_train, val_dropped = load_data(train_path, val_path, missing_threshold, replacement_ratio)

		# Convert to LSTM format
		train_X, train_y = LSTM_format(upsampled_train, num_of_past_hours, extra_padding)
		val_X, val_y = LSTM_format(val_dropped, num_of_past_hours, extra_padding)

		# Train
		model = Model_Architecture(train_X,
				rnn_type = rnn_type,
				rnn_layers = rnn_layers,
				dense_layers = dense_layers,
				rnn_layer_size = rnn_layer_size,
				dense_layer_size = dense_layer_size, 
				rnn_dropout = rnn_dropout, 
				dense_dropout = dense_dropout)

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
				callbacks=[ES, tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR)])

		# get model accuracy
		normalized_score, utility_score, ACC, SN, PPV, F1, AP, AUROC = evaluate_model(model, val_X, val_y, val_dropped, num_of_past_hours, extra_padding)
		LOG.debug('Final sepsis score is: %d', normalized_score)
		nni.report_final_result({'default': normalized_score, "utility_score": utility_score, "accuracy":ACC, 
				"sensitivity": float(np.nan_to_num(SN)), "precision": np.nan_to_num(PPV), "f1_score": np.nan_to_num(F1), 
				"AUPRC": np.nan_to_num(AP), "AUROC": np.nan_to_num(AUROC)})

	except Exception as e:
		LOG.exception(e)
		raise
