import tensorflow as tf
sess = tf.Session()
from tensorflow import gfile
from tensorflow.python.framework import graph_util
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from keras import backend as K
K.set_session(sess)
K.set_learning_phase(0)
from keras.models import load_model
from keras.applications.mobilenet import DepthwiseConv2D
from keras.activations import softmax
from model import relu6, overlapping_time_slice_stack
from classes import get_classes
from utils import smooth_categorical_crossentropy

DATA_TENSOR_NAME = 'decoded_sample_data'
FINAL_TENSOR_NAME = 'labels_softmax'
FROZEN_PATH = 'tf_files/frozen_206.pb'
# OPTIMIZED_PATH = 'tf_files/optimized_206.pb'
wanted_classes = get_classes(wanted_only=True)
all_classes = get_classes(wanted_only=False)
custom_objects = {
    'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
    'overlapping_time_slice_stack': overlapping_time_slice_stack,
    'softmax': softmax, '<lambda>': smooth_categorical_crossentropy}


model = load_model('checkpoints_206/ep-064-vl-0.2328.hdf5',
                   custom_objects=custom_objects)

# rename placeholders for special prize:
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge#Prizes
# decoded_sample_data:0, taking a [16000, 1] float tensor as input,
# representing the audio PCM-encoded data.
# `decode_wav` will produce two outputs. tf names them: 'name:0', 'name:1'.
wav_filename_placeholder_ = tf.placeholder(tf.string, [], name='wav_fn')
wav_loader = io_ops.read_file(wav_filename_placeholder_)
wav_decoder = contrib_audio.decode_wav(
    wav_loader, desired_channels=1, desired_samples=16000,
    name=DATA_TENSOR_NAME)

# add batch dimension and remove last one
# keras model wants (None, 16000)
data_reshaped = tf.reshape(wav_decoder.audio, (1, -1))
# call keras model
softmax_probs = model(data_reshaped)
# remove batch dimension
softmax_probs = tf.reshape(
    softmax_probs, (-1, ), name=FINAL_TENSOR_NAME)

frozen_graph_def = graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(),
    [FINAL_TENSOR_NAME])

with gfile.FastGFile(FROZEN_PATH, 'wb') as f:
  f.write(frozen_graph_def.SerializeToString())

print("Wrote frozen graph to: %s" % FROZEN_PATH)
