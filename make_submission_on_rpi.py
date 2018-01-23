import tensorflow as tf
# comment December 11, 2017 at 12:11 am:
# https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi
try:
  from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio  # noqa
except ImportError:
  from tensorflow.python.ops.gen_audio_ops import *  # noqa
from scipy.io import wavfile as wf
from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def main():
  # requirements:
  # https://www.kaggle.com/c/tensorflow-speech-recognition-challenge#Prizes
  batch_size = 1  # batch dimension should be 1
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--frozen_graph',
      type=str,
      default='tf_files/frozen.pb',
      help="""\
      Path to frozen graph.\
      """)
  parser.add_argument(
      '--data_tensor',
      type=str,
      default='decoded_sample_data:0',
      help="""\
      Input data tensor name. Leave as is for the
      competition.\
      """)
  parser.add_argument(
      '--rate_tensor',
      type=str,
      default='decoded_sample_data:1',
      help="""\
      Input rate tensor name. Leave as is for the
      competition.\
      """)
  parser.add_argument(
      '--output_tensor',
      type=str,
      default='labels_softmax:0',
      help="""\
      Name of the softmax output tensor. Leave as is for the
      competition.\
      """)
  parser.add_argument(
      '--test_data',
      type=str,
      default='data/test/audio',
      help="""\
      Path to the test wav files directory.\
      """)
  parser.add_argument(
      '--submission_fn',
      type=str,
      default='rpi_submission.csv',
      help="""\
      The submission's filename.\
      """)

  args, unparsed = parser.parse_known_args()
  test_fns = sorted(glob(os.path.join(args.test_data, '*.wav')))
  sess = tf.Session()
  # sample_rate = 16000
  classes = '_silence_ _unknown_ stop down off right up go on yes left no'.split() # noqa
  int2label = {i: c for i, c in enumerate(classes)}
  load_graph(args.frozen_graph)
  data_tensor = sess.graph.get_tensor_by_name(args.data_tensor)
  rate_tensor = sess.graph.get_tensor_by_name(args.rate_tensor)
  output_tensor = sess.graph.get_tensor_by_name(args.output_tensor)

  fns, wanted_labels, probabilities = [], [], []
  batch_counter = 0
  X_batch = []

  for i in tqdm(range(len(test_fns[:]))):
    test_fn = test_fns[i]
    fns.append(os.path.basename(test_fn))
    rate, wav_data = wf.read(test_fn)
    # assert rate == 16000
    wav_data = np.float32(wav_data) / 32767
    # assert len(wav_data) == 16000
    wav_data = wav_data.reshape((-1, 1))
    X_batch.append(wav_data)

    batch_counter += 1
    if batch_counter == batch_size:
      probs = sess.run(
          output_tensor, {data_tensor: wav_data, rate_tensor: rate})
      pred = probs.argmax(axis=-1)
      probabilities.append(probs)
      pred_label = int2label[int(pred)]
      # _unknown_ to unknown and _silence_ to silence
      pred_label = pred_label.strip('_')
      wanted_labels.append(pred_label)
      X_batch = []
      batch_counter = 0

  pd.DataFrame({'fname': fns, 'label': wanted_labels}).to_csv(
      args.submission_fn,
      index=False, compression=None)


if __name__ == '__main__':
  main()
