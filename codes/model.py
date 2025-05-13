import keras.backend as K
from keras.layers import Lambda
from keras.layers import *
from keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional, LSTM
from keras.regularizers import l1, l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# Channel Attention Module
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(
        input_feature)  # GlobalAveragePooling2D Only two dimensions remain: batchsize and channel. Shape: [B,H,W,C] → [B,C]
    avg_pool = Reshape((1, 1, channel))(avg_pool)  # Change shape: width, height, depth (pull into a vector so you can feed MLP)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(
        input_feature)  # GlobalMaxPooling2D There are only two dimensions left: batchsize and channel. Shape: [B,H,W,C] → [B,C]
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool, max_pool])  # The results of the treatment are summed
    cbam_feature = Activation('sigmoid')(cbam_feature)  # Get a weight map for each channel

    return multiply([input_feature, cbam_feature])


# Spatial attention model

def spatial_attention(input_feature):
    kernel_size = 6

    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)  # Average the tensors, change the third-dimensional coordinates, and keep the original dimension
    # assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


# CBAM
def cbam_block(cbam_feature, ratio=8):
    channel_feature = channel_attention(cbam_feature, ratio)
    spatial_feature = spatial_attention(cbam_feature)
    return channel_feature, spatial_feature


def build_model_2(dropout_rate=0.15, weight_decay=0):
    promoters = Input(shape=(10, 81, 1))

    x_2 = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(1, 1),
                 use_bias=False, padding='same', kernel_regularizer=l2(weight_decay))(promoters)
    x_2 = BatchNormalization()(x_2)
    x_2 = Dropout(dropout_rate)(x_2)
    x_2 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(1, 1),
                 use_bias=False, padding='same', kernel_regularizer=l2(weight_decay))(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = Dropout(dropout_rate)(x_2)

    # Adding an attention module layer
    channel_feature, spatial_feature = cbam_block(x_2)
    # mathematics
    x_2 = multiply([channel_feature, spatial_feature])

    x_2 = Flatten()(x_2)

    input_2 = Input(shape=(27, 640))
    x_1 = Dense(256, activation='relu')(input_2)
    x_1 = Dropout(0.4)(x_1)

    x_1 = Dense(128, activation='relu')(x_1)
    x_1 = Dropout(0.4)(x_1)

    x_1 = Dense(64, activation='relu')(x_1)
    x_1 = Dropout(0.4)(x_1)

    x_1 = Flatten()(x_1)

    x = Concatenate(axis=1)([x_2, x_1])

    x = Dense(units=240, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(0.3)(x)

    x = Dense(units=40, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    x = Dropout(0.3)(x)

    x = Dense(units=2, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    outputs = [x]

    model = Model([promoters, input_2], outputs)
    optimizer = Adam(learning_rate=2.2e-4, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
