# =========================================================================
#   (c) Copyright 2020
#   All rights reserved
#   Programs written by Hao Liu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pandas as pd
from sklearn.utils import class_weight

# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import *
from keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

sys.stderr = stderr

import numpy as np
import sys
import csv

import tensorflow as tf
from tensorflow.keras.layers import Input, LayerNormalization, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Conv1D, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import AdamW

import numpy as np

try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    pass

def get_df_values(type, time_window, df_values0):
    if type == 'patchTST':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 20, 7, 15, 8, 21, 6, 18, 5, 10, 9, 17, 16, 19, 12, 14, 4]]  # 12 patchTST
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 20, 11, 13, 9, 15, 14, 8, 7, 5, 21, 6, 17, 18, 10, 12, 16, 4, 19]]  # 24 patchTST 
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 20, 13, 5, 14, 8, 15, 7, 9, 21, 6, 4, 12, 17, 18, 10, 16, 19]]  # 36 patchTST
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 20, 13, 9, 14, 7, 15, 8, 6, 4, 21, 12, 17, 18, 16, 10, 19]]  # 48  patchTST
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60 patchTST
        return df_values

def load_data(datafile, series_len, start_feature, n_features, mask_value, type, time_window):
    df = pd.read_csv(datafile, header=None)
    df_values0 = df.values
    df_values = get_df_values(type, time_window, df_values0)
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    n_neg = 0
    n_pos = 0
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        label = row[0]
        if label == 'padding':
            continue
        has_zero_record = False
        # if one of the physical feature values is missing, then discard it.
        for k in range(start_feature, start_feature + n_features):
            if float(row[k]) == 0.0:
                has_zero_record = True
                break

        if has_zero_record is False:
            cur_harp_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_harp_num = int(prev_row[3])
                if prev_harp_num != cur_harp_num:
                    break
                has_zero_record_tmp = False
                for k in range(start_feature, start_feature + n_features):
                    if float(prev_row[k]) == 0.0:
                        has_zero_record_tmp = True
                        break
                if float(prev_row[-5]) >= 3500 or float(prev_row[-4]) >= 65536 or \
                        abs(float(prev_row[-1]) - float(prev_row[-2])) > 70:
                    has_zero_record_tmp = True

                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if (label == 'N' or label == 'P') and len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                if label == 'N':
                    y.append(0)
                    n_neg += 1
                elif label == 'P':
                    y.append(1)
                    n_pos += 1
    X_arr = np.array(X)
    y_arr = np.array(y)
    nb = n_neg + n_pos
    return X_arr, y_arr, nb


def attention_3d_block(hidden_states, series_len):
    hidden_size = int(hidden_states.shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
    hidden_states_t = Reshape((hidden_size, series_len), name='attention_input_reshape')(hidden_states_t)
    score_first_part = Dense(series_len, use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_patches, d_model, **kwargs):
        super().__init__(**kwargs)
        self.positional_encoding = self.get_positional_encoding(num_patches, d_model)

    def get_positional_encoding(self, num_patches, d_model):
        positions = np.arange(num_patches)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((num_patches, d_model))
        pos_encoding[:, 0::2] = np.sin(positions * div_term)
        pos_encoding[:, 1::2] = np.cos(positions * div_term)
        return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding
    
# Stochastic Depth (Random Layer Dropping)
def stochastic_depth(x, survival_prob=0.9):
    return tf.keras.layers.Dropout(1 - survival_prob)(x) if tf.random.uniform([]) < survival_prob else x

# PatchTST Model
def patchTST(n_features, series_len, patch_size=4, num_heads=10, ff_dim=512, dropout_rate=0.1, num_transformer_blocks=6):
    inputs = Input(shape=(series_len, n_features))
    num_patches = series_len // patch_size

    # Patch Embedding (Conv1D)
    x = Conv1D(filters=patch_size * n_features, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    x = Reshape((num_patches, patch_size * n_features))(x)
    x = LayerNormalization()(x)

    # Add Positional Encoding
    x = PositionalEncoding(num_patches, patch_size * n_features)(x)

    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-Head Attention with increased key_dim
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = LayerNormalization()(x + attn_output)  # Residual Connection

        # Feed Forward Network
        ff_output = Dense(ff_dim, activation='relu', kernel_initializer='he_normal')(x)
        ff_output = BatchNormalization()(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(x.shape[-1], activation='relu')(ff_output)
        
        # Apply Residual Connection
        x = LayerNormalization()(x + ff_output)

        # Apply Stochastic Depth for Regularization
        x = stochastic_depth(x, survival_prob=0.9)

    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Final Dense Layer for Binary Classification
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    # Compile the model (SAME as your training script)
    model = Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.001, weight_decay=0.01), metrics=['accuracy'])

    return model


def output_result(test_data_file, result_file, type, time_window, start_feature, n_features, thresh):
    df = pd.read_csv(test_data_file, header=None)
    df_values0 = df.values
    df_values = get_df_values(type, time_window, df_values0)
    with open(result_file, 'w', encoding='UTF-8') as result_csv:
        w = csv.writer(result_csv)
        w.writerow(['Predicted Label', 'Label', 'Timestamp', 'NOAA AR NUM', 'HARP NUM',
                      'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'AREA_ACR',
                      'MEANPOT', 'R_VALUE', 'SHRGT45', 'MEANGAM', 'MEANJZH', 'MEANGBT', 'MEANGBZ',
                      'MEANJZD', 'MEANGBH', 'MEANSHR', 'MEANALP'])
        idx = 0
        for i in range(len(df_values)):
            line = df_values[i].tolist()
            if line[0] == 'padding' or float(line[-5]) >= 3500 or float(line[-4]) >= 65536 \
                    or abs(float(line[-1]) - float(line[-2])) > 70:
                continue
            has_zero_record = False
            # if one of the physical feature values is missing, then discard it.
            for k in range(start_feature, start_feature + n_features):
                if float(line[k]) == 0.0:
                    has_zero_record = True
                    break
            if has_zero_record:
                continue
            if prob[idx] >= thresh:
                line.insert(0, 'P')
            else:
                line.insert(0, 'N')
            idx += 1
            w.writerow(line)
            
def get_n_features_thresh(type, time_window):
    n_features = 0
    thresh = 0
    if type == 'patchTST':
        if time_window == 12:
            n_features = 15
            thresh = 0.4
        elif time_window == 24:
            n_features = 18
            thresh = 0.45
        elif time_window == 36:
            n_features = 8
            thresh = 0.45
        elif time_window == 48:
            n_features = 15
            thresh = 0.45
        elif time_window == 60:
            n_features = 6
            thresh = 0.5
    return n_features, thresh

if __name__ == '__main__':
    type = sys.argv[1]
    time_window = int(sys.argv[2])
    train_again = int(sys.argv[3])
    train_data_file = './normalized_training_' + str(time_window) + '.csv'
    test_data_file = './normalized_testing_' + str(time_window) + '.csv'
    result_file = './' + type + '-' + str(time_window) + '-output.csv'
    model_file = './' + type + '-' + str(time_window) + '-model.h5'
    start_feature = 4
    n_features, thresh = get_n_features_thresh(type, time_window)
    mask_value = 0
    series_len = 20
    epochs = 20
    batch_size = 256
    nclass = 2

    if train_again == 1:
        # Train
        print('loading training data...')
        X_train, y_train, nb_train = load_data(datafile=train_data_file,
                                               series_len=series_len,
                                               start_feature=start_feature,
                                               n_features=n_features,
                                               mask_value=mask_value,
                                               type=type,
                                               time_window=time_window)


        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_ = {0: class_weights[0], 1: class_weights[1]}
        print('done loading training data...')

        model = patchTST(n_features, series_len)
        
        print('training the model, wait until it is finished...')
        model.compile(loss='binary_crossentropy',
                      optimizer='RMSprop',
                      metrics=['accuracy'])

        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=False,
                            shuffle=True,
                            class_weight=class_weight_)
        print('finished...')
        model.save(model_file)
    else:
        print('loading model...')
        model = load_model(model_file,custom_objects={"PositionalEncoding": PositionalEncoding})
        print('done loading...')

    # Test
    print('loading testing data')
    X_test, y_test, nb_test = load_data(datafile=test_data_file,
                                        series_len=series_len,
                                        start_feature=start_feature,
                                        n_features=n_features,
                                        mask_value=mask_value,
                                        type=type,
                                        time_window=time_window)
    print('done loading testing data...')
    print('predicting testing data...')
    prob = model.predict(X_test,
                         batch_size=batch_size,
                         verbose=False,
                         steps=None)
    print('done predicting...')
    print('writing prediction results into file...')
    output_result(test_data_file=test_data_file,
                  result_file=result_file,
                  type=type,
                  time_window=time_window,
                  start_feature=start_feature,
                  n_features=n_features,
                  thresh=thresh)
    print('done...')

import numpy as np    
from sklearn.metrics import accuracy_score
y_pred = (prob >= 0.5).astype(int)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")


#TSS scores 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np

def compute_tss(y_true, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    return tpr - fpr

def compute_fixed_threshold_tss(y_true, y_pred_prob, fixed_thresh):
    y_pred_binary = (y_pred_prob >= fixed_thresh).astype(int)
    return compute_tss(y_true, y_pred_binary)

# Predict with Fixed Threshold 

# Predict on testing data
baseline_preds = model.predict(X_test)

# Set a fixed threshold (you can adjust this manually if you want)
fixed_threshold = thresh  # <-- used the same threshold tahts was set earlier(0.45)

# Compute baseline TSS
baseline_tss = compute_fixed_threshold_tss(y_test, baseline_preds, fixed_threshold)

print(f"\n Baseline TSS: {baseline_tss:.4f} at Threshold: {fixed_threshold:.2f}")

# Fixed Threshold Feature Importance

# List your full feature names
all_feature_names = [
    'MEANPOT', 'SHRGT45', 'TOTPOT', 'USFLUX', 'MEANJZH',
    'ABSNJZH', 'SAVNCPP', 'MEANALP', 'MEANSHR', 'TOTUSJZ',
    'TOTUSJH', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'AREA_ACR',
    'R_VALUE', 'MEANGBT', 'MEANGBH'
]

_, _, num_features = X_test.shape
feature_names = all_feature_names[:num_features]

feature_fixed_tss_scores = {}

for i, feature in enumerate(feature_names):
    print(f"\nMasking feature: {feature}")

    X_ablate = np.array(X_test, copy=True)
    X_ablate[:, :, i] = 0  # Mask the i-th feature

    preds_ablate = model.predict(X_ablate)
    ablate_tss = compute_fixed_threshold_tss(y_test, preds_ablate, fixed_threshold)

    feature_fixed_tss_scores[feature] = ablate_tss

    print(f"TSS with {feature} masked (fixed threshold {fixed_threshold:.2f}): {ablate_tss:.4f}")

# Ranking Features

features_sorted = sorted(feature_fixed_tss_scores.items(), key=lambda x: x[1], reverse=True)

print("\n Final Feature Ranking Based on Fixed-Threshold TSS ")
for rank, (feat, tss_score) in enumerate(features_sorted, 1):
    print(f"{rank:2d}. {feat:10s} --> TSS = {tss_score:.4f}")



