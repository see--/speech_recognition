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
from utils import smooth_categorical_crossentropy
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_tensor',
      type=str,
      default='decoded_sample_data',
      help="""\
      Input data tensor name. Leave as is for the
      competition.\
      """)
  parser.add_argument(
      '--final_tensor',
      type=str,
      default='labels_softmax',
      help="""\
      Name of the softmax output tensor. Leave as is for the
      competition.\
      """)
  parser.add_argument(
      '--frozen_path',
      type=str,
      default='tf_files/frozen.pb',
      help="""\
      The frozen graph's filename.\
      """)
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='checkpoints_106/ep-062-vl-0.1815.hdf5',
      help="""\
      Path to the hdf5 checkpoint that you want to freeze.\
      """)
  args, unparsed = parser.parse_known_args()
  custom_objects = {
      'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
      'overlapping_time_slice_stack': overlapping_time_slice_stack,
      'softmax': softmax, '<lambda>': smooth_categorical_crossentropy}

  model = load_model(args.checkpoint_path,
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
      name=args.data_tensor)

  # add batch dimension and remove last one
  # keras model wants (None, 16000)
  data_reshaped = tf.reshape(wav_decoder.audio, (1, -1))
  # call keras model
  softmax_probs = model(data_reshaped)
  # remove batch dimension
  softmax_probs = tf.reshape(
      softmax_probs, (-1, ), name=args.final_tensor)

  frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph.as_graph_def(),
      [args.final_tensor])

  with gfile.FastGFile(args.frozen_path, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())

  print("Wrote frozen graph to: %s" % args.frozen_path)


if __name__ == '__main__':
  main()
