import keras
from keras.layers import Dense, Input, Lambda
from keras.layers.noise import AlphaDropout
from keras.models import Model
import tensorflow as tf


def preprocess(x):
  x = (x + 0.8) / 7.0
  x = tf.clip_by_value(x, -5, 5)
  return x


Preprocess = Lambda(preprocess)


def snn_model(input_size=16000, num_classes=11):
  input_layer = Input(shape=[input_size])
  activation = 'selu'
  kernel_initializer = 'lecun_normal'
  x = input_layer
  x = Preprocess(x)
  for num_hidden, dropout_ratio in [
          (512, 0.1), (256, 0.1), (128, 0.1), (64, 0.05)]:
      x = Dense(num_hidden, activation=activation,
                kernel_initializer=kernel_initializer)(x)
      x = AlphaDropout(dropout_ratio)(x)

  x = Dense(num_classes, activation='softmax',
            kernel_initializer=kernel_initializer)(x)

  model = Model(input_layer, x, name='speech_model')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.01 / 2, momentum=0.9),
      loss=keras.losses.sparse_categorical_crossentropy,
      metrics=[keras.metrics.sparse_categorical_accuracy])
  return model


def simple_model(input_size=16000, num_classes=11):
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Preprocess(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='speech_model')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
      loss=keras.losses.sparse_categorical_crossentropy,
      metrics=[keras.metrics.sparse_categorical_accuracy])
  return model


def speech_model(model_type, input_size, num_classes=11):
  if model_type == 'simple':
    return simple_model(input_size, num_classes)
  elif model_type == 'snn':
    return snn_model(input_size, num_classes)
  else:
    raise ValueError("Invalid model")
