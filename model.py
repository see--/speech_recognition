import keras
from keras import backend as K
from keras.layers import Dense, Input, Lambda, Conv1D, AveragePooling1D
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPool1D
from keras.layers import BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, MaxPool1D
from keras.layers import Dropout, Add, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Concatenate, AveragePooling2D
from keras.layers.noise import AlphaDropout
from keras.regularizers import l2
from keras.models import Model


def preprocess(x):
  x = (x + 0.8) / 7.0
  x = K.clip(x, -5, 5)
  return x


def preprocess_raw(x):
  x = K.pow(10.0, x) - 1.0
  return x


Preprocess = Lambda(preprocess)


PreprocessRaw = Lambda(preprocess_raw)


Relu6 = Lambda(lambda x: K.relu(x, max_value=6))


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
      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
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


def conv_1d_simple_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = PreprocessRaw(x)
  x = Reshape([-1, 1])(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
    x = Conv1D(num_filters, k, padding=padding, strides=strides,
               kernel_regularizer=l2(0.00001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
    x = Conv1D(num_filters, k, padding=padding, dilation_rate=dilation_rate,
               kernel_regularizer=l2(0.00001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  x = _reduce_conv(x, 32, 5, strides=4)  # 4000
  x = _context_conv(x, 48, 3)
  x = _reduce_conv(x, 64, 3)  # 2000
  x = _context_conv(x, 48, 1)
  x = _context_conv(x, 72, 3)
  x = _reduce_conv(x, 72, 3)  # 1000
  x = _context_conv(x, 64, 1)
  x = _context_conv(x, 96, 3)
  x = _reduce_conv(x, 96, 3)  # 500
  x = _context_conv(x, 72, 1)
  x = _context_conv(x, 128, 3)
  x = _reduce_conv(x, 128, 3)  # 250
  x = _context_conv(x, 96, 1)
  x = _context_conv(x, 160, 3)
  x = _reduce_conv(x, 160, 3)  # 125
  x = _context_conv(x, 128, 1)
  x = _context_conv(x, 192, 3)
  x = _reduce_conv(x, 192, 3)  # 64
  x = _context_conv(x, 160, 1)
  x = _context_conv(x, 256, 3)
  x = _reduce_conv(x, 256, 3)  # 64
  x = _context_conv(x, 192, 1)
  x = _context_conv(x, 288, 3)
  x = _reduce_conv(x, 288, 3)  # 64
  x = _context_conv(x, 256, 1)
  x = _context_conv(x, 288, 3)

  x = Bidirectional(GRU(144, dropout=0.1, recurrent_dropout=0.1))(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='conv_1d_time_stacked')
  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_1d_inception_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = PreprocessRaw(x)
  x = Reshape([-1, 1])(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='same'):
    x = Conv1D(num_filters, k, padding=padding, strides=strides,
               kernel_regularizer=l2(0.00001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='same'):
    x = Conv1D(num_filters, k, padding=padding, dilation_rate=dilation_rate,
               kernel_regularizer=l2(0.00001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  def _stem(x):
    x = _reduce_conv(x, 32, 5, strides=4, padding='valid')  # ~4000
    x = _context_conv(x, 32, 3, padding='valid')
    x = _reduce_conv(x, 64, 3, padding='valid')  # ~2000
    x = _context_conv(x, 64, 3, padding='valid')
    x = _reduce_conv(x, 128, 3, padding='valid')  # ~1000
    x = _context_conv(x, 128, 3, padding='valid')
    x = _reduce_conv(x, 256, 3, padding='valid')  # ~500
    x = _context_conv(x, 256, 3, padding='valid')
    x = _reduce_conv(x, 384, 3, padding='valid')  # ~250
    x = _context_conv(x, 384, 3, padding='valid')
    x = _reduce_conv(x, 512, 3, padding='valid')  # ~125
    x = _context_conv(x, 512, 3, padding='valid')
    return x

  def _inception_block(x, base_num, block_id):
    branch1x1 = _context_conv(x, int(2 * base_num), 1)

    branch5x5 = _context_conv(x, int(1.5 * base_num), 1)
    branch5x5 = _context_conv(branch5x5, int(2 * base_num), 5)

    branch3x3dbl = _context_conv(x, int(2 * base_num), 1)
    branch3x3dbl = _context_conv(branch3x3dbl, int(3 * base_num), 3)
    branch3x3dbl = _context_conv(branch3x3dbl, int(3 * base_num), 3)

    branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = _context_conv(branch_pool, base_num, 1)
    x = Concatenate(name='mixed%d' % block_id)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])
    return x

  def _reduce_inception_block(x, base_num, strides, block_id):
    branch3x3 = _reduce_conv(x, int(6 * base_num), 3, strides=strides,
                             padding='valid')

    branch3x3dbl = _context_conv(x, base_num, 1)
    branch3x3dbl = _context_conv(branch3x3dbl, int(1.5 * base_num), 3)
    branch3x3dbl = _reduce_conv(branch3x3dbl, int(1.5 * base_num), 3,
                                strides=strides, padding='valid')

    branch_pool = MaxPool1D(3, strides=strides)(x)
    x = Concatenate(name='mixed%d' % block_id)(
        [branch3x3, branch3x3dbl, branch_pool])
    return x

  # from IPython import embed
  # embed()
  x = _stem(x)
  x = _inception_block(x, base_num=32, block_id=1)
  x = _inception_block(x, base_num=16, block_id=2)
  x = _reduce_inception_block(x, base_num=32, strides=2, block_id=3)
  x = _inception_block(x, base_num=32, block_id=4)
  x = _inception_block(x, base_num=32, block_id=5)
  x = _reduce_inception_block(x, base_num=64, strides=2, block_id=6)
  x = _inception_block(x, base_num=64, block_id=7)
  x = _inception_block(x, base_num=64, block_id=8)
  x = _reduce_inception_block(x, base_num=96, strides=2, block_id=9)
  x = _inception_block(x, base_num=96, block_id=10)
  x = _inception_block(x, base_num=96, block_id=11)

  x = Dropout(0.15)(x)
  x = Conv1D(num_classes, 14, activation='softmax', padding='valid')(x)
  x = Reshape([-1])(x)

  model = Model(input_layer, x, name='conv_1d_time_stacked')
  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_1d_time_stacked_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([400, 40])(x)
  x = PreprocessRaw(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
    x = Conv1D(num_filters, k, padding=padding, use_bias=False,
               kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
    x = Conv1D(num_filters, k, padding=padding, dilation_rate=dilation_rate,
               kernel_regularizer=l2(0.00001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  x = _context_conv(x, 64, 1)
  x = _reduce_conv(x, 96, 3)
  x = _context_conv(x, 96, 3)
  x = _reduce_conv(x, 128, 3)
  x = _context_conv(x, 128, 3)
  x = _reduce_conv(x, 160, 3)
  x = _context_conv(x, 160, 3)
  x = _reduce_conv(x, 192, 3)
  x = _context_conv(x, 192, 3)
  x = _reduce_conv(x, 224, 3)
  x = _context_conv(x, 224, 3)

  x = Dropout(0.2)(x)
  x = Conv1D(num_classes, 5, activation='softmax')(x)
  x = Reshape([-1])(x)

  model = Model(input_layer, x, name='conv_1d_time_stacked')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.96),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_inception_d1_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([400, 40])(x)
  x = PreprocessRaw(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='same'):
    x = Conv1D(num_filters, k, padding=padding, use_bias=False,
               kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='same'):
    x = Conv1D(num_filters, k, padding=padding, dilation_rate=dilation_rate,
               kernel_regularizer=l2(0.00001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  def _inception_block(x, base_num, block_id):
    branch1x1 = _context_conv(x, int(2 * base_num), 1)

    branch5x5 = _context_conv(x, int(1.5 * base_num), 1)
    branch5x5 = _context_conv(branch5x5, int(2 * base_num), 3, dilation_rate=2)

    branch3x3dbl = _context_conv(x, int(2 * base_num), 1)
    branch3x3dbl = _context_conv(
        branch3x3dbl, int(3 * base_num), 3, dilation_rate=2)
    branch3x3dbl = _context_conv(
        branch3x3dbl, int(3 * base_num), 3, dilation_rate=2)

    branch_pool = AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    branch_pool = _context_conv(branch_pool, base_num, 1)
    x = Concatenate(name='mixed%d' % block_id)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])
    return x

  def _reduce_inception_block(x, base_num, strides, block_id):
    branch3x3 = _reduce_conv(x, int(6 * base_num), 3, strides=strides)

    branch3x3dbl = _context_conv(x, base_num, 1)
    branch3x3dbl = _context_conv(branch3x3dbl, int(1.5 * base_num), 3)
    branch3x3dbl = _reduce_conv(branch3x3dbl, int(1.5 * base_num), 3,
                                strides=strides, padding='same')

    branch_pool = MaxPool1D(3, strides=strides, padding='same')(x)
    x = Concatenate(name='mixed%d' % block_id)(
        [branch3x3, branch3x3dbl, branch_pool])
    return x
  # stem: output @ ~100
  x = _context_conv(x, 64, 1)
  x = _reduce_conv(x, 96, 3, padding='valid')
  x = _context_conv(x, 96, 3, padding='valid')
  x = _reduce_conv(x, 128, 3, padding='valid')
  # inception block 1: output @ ~50
  x = _inception_block(x, base_num=16, block_id=1)
  x = _inception_block(x, base_num=16, block_id=2)
  x = _reduce_inception_block(x, base_num=32, strides=2, block_id=3)
  # inception block 2: @ ~24
  x = _inception_block(x, base_num=32, block_id=4)
  x = _inception_block(x, base_num=32, block_id=5)
  x = _reduce_inception_block(x, base_num=32, strides=2, block_id=6)
  # inception block 3: @ ~12
  x = _inception_block(x, base_num=32, block_id=7)
  x = _inception_block(x, base_num=32, block_id=8)
  x = _reduce_inception_block(x, base_num=32, strides=2, block_id=9)
  # inception block 4: @ ~6
  x = _inception_block(x, base_num=32, block_id=10)
  x = _inception_block(x, base_num=32, block_id=11)
  x = _reduce_inception_block(x, base_num=32, strides=2, block_id=12)

  x = Dropout(0.2)(x)
  x = Conv1D(num_classes, 6, activation='softmax')(x)
  x = Reshape([-1])(x)

  model = Model(input_layer, x, name='inception_d1')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.96),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def mobilenet_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([200, 80], name='time_stack')(x)
  x = PreprocessRaw(x)

  def _conv_1d_bn(x, num_filters, k, strides=1, padding='valid', l2_reg=1e-5):
    x = Conv1D(num_filters, k, padding=padding, strides=strides,
               kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  x = _conv_1d_bn(x, 100, 1)
  x = _conv_1d_bn(x, 100, 3, strides=2, padding='same')
  x = Reshape([100, 100, 1], name='spongebob')(x)
  from keras.applications import MobileNet  # noqa
  model = MobileNet(input_tensor=x, weights=None, classes=num_classes)
  model.name = "conv_1d_mobilenet"
  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_1d_gru_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([400, 40])(x)
  x = PreprocessRaw(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
    x = Conv1D(num_filters, k, padding=padding,
               kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
    x = Conv1D(num_filters, k, padding=padding, dilation_rate=dilation_rate,
               kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  x = _context_conv(x, 32, 1)
  x = _reduce_conv(x, 64, 3)
  x = _context_conv(x, 64, 3)
  x = _context_conv(x, 64, 3)
  x = _reduce_conv(x, 128, 3)
  x = _context_conv(x, 128, 3)
  x = _context_conv(x, 128, 3)
  x = _reduce_conv(x, 256, 3)
  x = _context_conv(x, 256, 3)
  x = _context_conv(x, 256, 3)
  x = _reduce_conv(x, 384, 3)
  x = _context_conv(x, 384, 3)
  x = _context_conv(x, 384, 3)  # (12, 384)

  x = Bidirectional(GRU(192, dropout=0.1, recurrent_dropout=0.1))(x)
  # x = Reshape([-1])(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='conv_1d_bigru')
  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_2d_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = True. This is the keras version of:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L165-L270  # noqa
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  # TODO(see--): Allow variable sized frequency_size & time_size
  frequency_size = 40
  time_size = 98
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([time_size, frequency_size, 1])(x)
  x = Preprocess(x)
  x = Conv2D(64, kernel_size=(20, 8), padding='same', activation='relu')(x)
  x = MaxPool2D()(x)
  x = Conv2D(128, kernel_size=(10, 4), padding='same', activation='relu')(x)
  x = MaxPool2D()(x)
  x = Flatten()(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='speech_model')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_2d_mobile_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = True. This is the keras version of:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L165-L270  # noqa
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  # TODO(see--): Allow variable sized frequency_size & time_size
  time_size = 98
  frequency_size = 40
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([time_size, frequency_size, 1])(x)
  x = Preprocess(x)

  def _conv_bn_relu6(x, num_filter, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(num_filter, kernel, padding='same',
               strides=strides)(x)
    x = BatchNormalization()(x)
    x = Relu6(x)
    return x

  x = _conv_bn_relu6(x, 32, strides=2)  # (49, 20)
  x = _conv_bn_relu6(x, 32)  # (49, 20)
  x = Dropout(0.05)(x)
  x = _conv_bn_relu6(x, 64, strides=2)  # (25, 10)
  x = _conv_bn_relu6(x, 64)  # (25, 10)
  x = Dropout(0.05)(x)
  x = _conv_bn_relu6(x, 128, strides=2)  # (13, 5)
  x = _conv_bn_relu6(x, 128)  # (13, 5)
  x = Dropout(0.05)(x)
  x = _conv_bn_relu6(x, 256, strides=2)  # (7, 3)
  x = _conv_bn_relu6(x, 256)  # (7, 3)
  x = Dropout(0.05)(x)

  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.1)(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='conv_2d_mobile')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.95),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def conv_2d_fast_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data. Note: Use only
  with compute_mfcc = True. This is the keras version of:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L165-L270  # noqa
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.
  Returns:
    Compiled keras model
  """
  # TODO(see--): Allow variable sized frequency_size & time_size
  time_size = 98
  frequency_size = 40
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([time_size, frequency_size, 1])(x)
  x = Preprocess(x)

  def _conv_bn(x, num_filter, kernel_size, dilation_rate):
    x = Conv2D(num_filter, kernel_size=kernel_size, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

  def _conv_bn_pool(x, num_filter, kernel_size, dilation_rate):
    x = _conv_bn(x, num_filter, kernel_size, dilation_rate)
    x = MaxPool2D()(x)
    return x

  x = _conv_bn_pool(x, 16, (11, 5), (2, 1))  # (49, 20)
  x = _conv_bn_pool(x, 32, (5, 3), (2, 1))  # (25, 10)
  x = _conv_bn_pool(x, 64, (3, 3), (1, 1))  # (13, 5)
  x = _conv_bn_pool(x, 128, (3, 3), (1, 1))  # (7, 3)
  x = GlobalAveragePooling2D()(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(input_layer, x, name='conv_2d_fast')
  model.compile(
      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def speech_model(model_type, input_size, num_classes=11):
  if model_type == 'simple':
    return simple_model(input_size, num_classes)
  elif model_type == 'snn':
    return snn_model(input_size, num_classes)
  elif model_type == 'conv_1d_time':
    return conv_1d_time_stacked_model(input_size, num_classes)
  elif model_type == 'conv_1d_simple':
    return conv_1d_simple_model(input_size, num_classes)
  elif model_type == 'conv_1d_gru':
    return conv_1d_gru_model(input_size, num_classes)
  elif model_type == 'conv_2d':
    return conv_2d_model(input_size, num_classes)
  elif model_type == 'conv_2d_fast':
    return conv_2d_fast_model(input_size, num_classes)
  elif model_type == 'conv_2d_mobile':
    return conv_2d_mobile_model(input_size, num_classes)
  elif model_type == 'mobilenet':
    return mobilenet_model(input_size, num_classes)
  elif model_type == 'inception':
    return conv_1d_inception_model(input_size, num_classes)
  elif model_type == 'inception_d1':
    return conv_inception_d1_model(input_size, num_classes)
  else:
    raise ValueError("Invalid model: %s" % model_type)


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
