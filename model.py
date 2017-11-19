import keras
from keras.layers import Dense, Input, Lambda, Conv1D, AveragePooling1D
from keras.layers import GlobalAveragePooling1D, Reshape, Flatten
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
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def simple_model(input_size=16000, num_classes=11):
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Preprocess(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='speech_model')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_1d_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([-1, 1])(x)
  x = Preprocess(x)

  def _reduce_conv(x, num_filters, k, strides=2):
    x = Conv1D(num_filters, k, padding='same', activation='relu')(x)
    x = AveragePooling1D(pool_size=strides)(x)
    return x

  def _context_conv(x, num_filters, dilation_rate):
    x = Conv1D(num_filters, 3, padding='same',
               dilation_rate=dilation_rate, activation='relu')(x)
    return x

  x = _reduce_conv(x, 8, 128, strides=4)
  x = _reduce_conv(x, 16, 64, strides=4)
  x = _reduce_conv(x, 32, 32)
  x = _reduce_conv(x, 64, 16)
  x = _context_conv(x, 1, 16)
  x = Flatten()(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='speech_model')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def speech_model(model_type, input_size, num_classes=11):
  if model_type == 'simple':
    return simple_model(input_size, num_classes)
  elif model_type == 'snn':
    return snn_model(input_size, num_classes)
  elif model_type == 'conv_1d_model':
    return conv_1d_model(input_size, num_classes)
  else:
    raise ValueError("Invalid model")


# from here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py  # noqa
def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.
  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.
  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }
