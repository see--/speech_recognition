from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from glob import glob
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from keras.models import load_model
from model import prepare_model_settings, relu6, overlapping_time_slice_stack
from keras.applications.mobilenet import DepthwiseConv2D
from keras.activations import softmax
from input_data import prepare_words_list
from classes import get_classes, get_int2label
from utils import center_crop
from IPython import embed  # noqa


def map_to_valid(labels):
    # map '_silence_' to 'silence'
    labels = [
        pl if pl != '_silence_' else 'silence' for pl in labels]
    # map '_unknown_' to 'unknown'
    labels = [
        pl if pl != '_unknown_' else 'unknown' for pl in labels]
    return labels


def map_to_wanted(labels, wanted_words):
  # map unknown words to 'unknown'
  labels = [
      pl if pl in wanted_words or pl == 'silence'
      else 'unknown' for pl in labels]
  return labels


if __name__ == '__main__':
  test_fns = sorted(glob('data/test/audio/*.wav'))
  tta_fns = sorted(glob('data/tta_test/audio/*.wav'))
  assert len(test_fns) == len(tta_fns)
  sess = K.get_session()
  K.set_learning_phase(0)
  sample_rate = 16000
  use_tta = False
  use_speed_tta = False
  wanted_only = False
  extend_reversed = False
  output_representation = 'mfcc'
  batch_size = 384
  wanted_words = prepare_words_list(get_classes(wanted_only=True))
  classes = get_classes(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  int2label = get_int2label(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  model_settings = prepare_model_settings(
      label_count=len(prepare_words_list(classes)), sample_rate=sample_rate,
      clip_duration_ms=1000, window_size_ms=30.0, window_stride_ms=15.0,
      dct_coefficient_count=60)

  wav_filename_placeholder = tf.placeholder(tf.string, [])
  wav_loader = io_ops.read_file(wav_filename_placeholder)
  wav_decoder = contrib_audio.decode_wav(
      wav_loader, desired_channels=1,
      desired_samples=model_settings['desired_samples'])
  clamped = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
  clamped = tf.reshape(clamped, (1, model_settings['desired_samples']))
  stfts = tf.contrib.signal.stft(
      clamped,
      frame_length=model_settings['window_size_samples'],
      frame_step=model_settings['window_stride_samples'],
      fft_length=None)
  spectrogram = tf.abs(stfts)
  num_spectrogram_bins = spectrogram.shape[-1].value
  lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
  linear_to_mel_weight_matrix = \
      tf.contrib.signal.linear_to_mel_weight_matrix(
          model_settings['dct_coefficient_count'],
          num_spectrogram_bins, model_settings['sample_rate'],
          lower_edge_hertz, upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
      spectrogram, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
  mfcc = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
      log_mel_spectrograms)[:, :]  # :13
  model = load_model('checkpoints_116/ep-033-vl-0.1529.hdf5',
                     custom_objects={'relu6': relu6,
                                     'DepthwiseConv2D': DepthwiseConv2D,
                                     'overlapping_time_slice_stack':
                                     overlapping_time_slice_stack,
                                     'softmax': softmax})
  # embed()

  # In wanted_labels we map the not wanted words to `unknown`. Though we
  # keep track of all labels in `labels`.
  fns, wanted_labels, labels, probabilities = [], [], [], []
  batch_counter = 0
  X_batch = []
  X_tta_batch = []
  for i in tqdm(range(len(test_fns[:]))):
    test_fn = test_fns[i]
    fns.append(os.path.basename(test_fn))
    if output_representation == 'mfcc':
      mfcc_val = sess.run(mfcc, {wav_filename_placeholder: test_fn})
      X_batch.append(mfcc_val.flatten())
    else:
      raw_val = sess.run(clamped, {wav_filename_placeholder: test_fn})
      X_batch.append(raw_val.flatten())
    if use_speed_tta:
      tta_fn = tta_fns[i]
      assert os.path.basename(tta_fn) == os.path.basename(test_fn)
      if output_representation == 'mfcc':
        mfcc_val = sess.run(mfcc, {wav_filename_placeholder: tta_fn})
        X_tta_batch.append(mfcc_val.flatten())
      else:
        raw_val = sess.run(clamped, {wav_filename_placeholder: tta_fn})
        X_tta_batch.append(raw_val.flatten())

    batch_counter += 1
    if batch_counter == batch_size:
      probs = model.predict(np.float32(X_batch))
      if use_tta:
        X_batch_left = np.float32(X_batch)
        X_batch_left = np.roll(X_batch_left, -1000, axis=1)
        left_probs = model.predict(X_batch_left)

        X_batch_left = np.float32(X_batch)
        X_batch_left = np.roll(X_batch_left, -2000, axis=1)
        left_probs2 = model.predict(X_batch_left)

        loud_probs = model.predict(1.2 * np.float32(X_batch))
        silent_probs = model.predict(0.8 * np.float32(X_batch))
        if use_speed_tta:
          slow_probs = model.predict(np.float32(X_tta_batch))
          probs = (probs + loud_probs + left_probs + slow_probs) / 4
        else:
          probs = (probs + loud_probs + silent_probs + left_probs +
                   left_probs2) / 5

      pred = probs.argmax(axis=-1)
      probabilities.append(probs)
      pred_labels = [int2label[int(p)] for p in pred]
      pred_labels = map_to_valid(pred_labels)
      labels.extend(pred_labels)

      pred_labels = map_to_wanted(pred_labels, wanted_words)
      wanted_labels.extend(pred_labels)
      # set back counter
      batch_counter, X_batch, X_tta_batch = 0, [], []

  # process remaining
  if X_batch:
    probs = model.predict(np.float32(X_batch))
    if use_tta:
      X_batch_left = np.float32(X_batch)
      X_batch_left = np.roll(X_batch_left, -1000, axis=1)
      left_probs = model.predict(X_batch_left)

      X_batch_left = np.float32(X_batch)
      X_batch_left = np.roll(X_batch_left, -2000, axis=1)
      left_probs2 = model.predict(X_batch_left)

      loud_probs = model.predict(1.2 * np.float32(X_batch))
      silent_probs = model.predict(0.8 * np.float32(X_batch))
      if use_speed_tta:
        slow_probs = model.predict(np.float32(X_tta_batch))
        probs = (probs + loud_probs + left_probs + slow_probs) / 4
      else:
        probs = (probs + loud_probs + silent_probs + left_probs +
                 left_probs2) / 5

    pred = probs.argmax(axis=-1)
    probabilities.append(probs)
    pred_labels = [int2label[int(p)] for p in pred]
    pred_labels = map_to_valid(pred_labels)
    labels.extend(pred_labels)

    pred_labels = map_to_wanted(pred_labels, wanted_words)
    wanted_labels.extend(pred_labels)

  pd.DataFrame({'fname': fns, 'label': wanted_labels}).to_csv(
      'submission_116.csv',
      index=False, compression=None)

  pd.DataFrame({'fname': fns, 'label': labels}).to_csv(
      'submission_116_all_labels.csv',
      index=False, compression=None)

  probabilities = np.concatenate(probabilities, axis=0)
  all_data = pd.DataFrame({'fname': fns, 'label': labels})
  for i, l in int2label.items():
    all_data[l] = probabilities[:, i]
  all_data.to_csv(
      'submission_116_all_labels_probs.csv',
      index=False, compression=None)
  print("Done!")
