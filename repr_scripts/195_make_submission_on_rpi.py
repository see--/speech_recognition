import tensorflow as tf
from scipy.io import wavfile
from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from input_data import prepare_words_list
from classes import get_classes, get_int2label


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


if __name__ == '__main__':
  # requirements:
  # https://www.kaggle.com/c/tensorflow-speech-recognition-challenge#Prizes
  frozen_graph_def = 'tf_files/frozen_195.pb'
  batch_size = 1  # batch dimension should be 1
  data_tensor_name = 'decoded_sample_data:0'
  rate_tensor_name = 'decoded_sample_data:1'
  output_tensor_name = 'labels_softmax:0'

  test_fns = sorted(glob('data/test/audio/*.wav'))
  sess = tf.Session()
  sample_rate = 16000
  use_tta = False
  wanted_only = True
  output_representation = 'raw'
  wanted_words = prepare_words_list(get_classes(wanted_only=True))
  classes = get_classes(wanted_only=wanted_only)
  int2label = get_int2label(wanted_only=wanted_only)
  load_graph(frozen_graph_def)
  data_tensor = sess.graph.get_tensor_by_name(data_tensor_name)
  rate_tensor = sess.graph.get_tensor_by_name(rate_tensor_name)
  output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

  fns, wanted_labels, probabilities = [], [], []
  batch_counter = 0
  X_batch = []

  for i in tqdm(range(len(test_fns[:]))):
    test_fn = test_fns[i]
    fns.append(os.path.basename(test_fn))
    rate, wav_data = wavfile.read(test_fn)
    # assert rate == 16000
    wav_data = np.float32(wav_data) / 32767
    # assert len(wav_data) == 16000
    wav_data = wav_data.reshape((-1, 1))
    X_batch.append(wav_data)

    batch_counter += 1
    if batch_counter == batch_size:
      probs = sess.run(
          output_tensor, {data_tensor: wav_data, rate_tensor: sample_rate})
      pred = probs.argmax(axis=-1)
      probabilities.append(probs)
      pred_label = int2label[int(pred)]
      # _unknown_ to unknown and _silence_ to silence
      pred_label = pred_label.strip('_')
      wanted_labels.append(pred_label)
      X_batch = []
      batch_counter = 0

  pd.DataFrame({'fname': fns, 'label': wanted_labels}).to_csv(
      'rpi_submission_195.csv',
      index=False, compression=None)
