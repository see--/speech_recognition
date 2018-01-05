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

FINAL_TENSOR_NAME = 'labels_softmax'
FROZEN_PATH = 'tf_files/frozen.pb'
OPTIMIZED_PATH = 'tf_files/optimized.pb'

wanted_classes = get_classes(wanted_only=True)
all_classes = get_classes(wanted_only=False)

model = load_model('checkpoints_177/ep-070-vl-0.2478.hdf5',
                   custom_objects={'relu6': relu6,
                                   'DepthwiseConv2D': DepthwiseConv2D,
                                   'overlapping_time_slice_stack':
                                   overlapping_time_slice_stack,
                                   'softmax': softmax,
                                   '<lambda>':
                                   smooth_categorical_crossentropy})

# rename placeholders for special prize:
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge#Prizes
# decoded_sample_data:0, taking a [16000, 1] float tensor as input,
# representing the audio PCM-encoded data.
wav_filename_placeholder_ = tf.placeholder(tf.string, [], name='wav_fn')
wav_loader = io_ops.read_file(wav_filename_placeholder_)
wav_decoder = contrib_audio.decode_wav(
    wav_loader, desired_channels=1, desired_samples=16000,
    name='decoded_sample_data')
# add batch dimension and remove last one
# keras model wants (None, 16000)
data_reshaped = tf.reshape(wav_decoder.audio, (1, -1))
# call keras model
all_probs = model(data_reshaped)
# remove batch dimension
all_probs = tf.reshape(all_probs, (-1, ))
# map classes to 12 wanted classes:
# 'silence unknown', 'stop down off right up go on yes left no'
# models were trained with 32 classes:
# 'silence unknown', 'sheila nine stop bed four six down bird marvin cat off right seven eight up three happy go zero on wow dog yes five one tree house two left no'  # noqa
mapped_classes = []
mapped_classes.append(all_probs[0])  # silence
mapped_classes.append(all_probs[1])  # unknown
for i, c in enumerate(all_classes):
  if c in wanted_classes:
    mapped_classes.append(all_probs[i + 2])
  else:
    mapped_classes[1] += all_probs[i + 2]

mapped_probs = tf.stack(mapped_classes, name=FINAL_TENSOR_NAME)

frozen_graph_def = graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(),
    [FINAL_TENSOR_NAME])

with gfile.FastGFile(FROZEN_PATH, 'wb') as f:
  f.write(frozen_graph_def.SerializeToString())

print("Wrote frozen graph to: %s" % FROZEN_PATH)
