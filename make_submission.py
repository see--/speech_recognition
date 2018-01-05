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
from input_data import prepare_words_list, AudioProcessor
from classes import get_classes, get_int2label
from utils import smooth_categorical_crossentropy
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
  sess = K.get_session()
  K.set_learning_phase(0)
  sample_rate = 16000
  use_tta = True
  use_speed_tta = False
  if use_speed_tta:
    tta_fns = sorted(glob('data/tta_test/audio/*.wav'))
    assert len(test_fns) == len(tta_fns)
  wanted_only = False
  extend_reversed = False
  output_representation = 'raw'
  batch_size = 384
  wanted_words = prepare_words_list(get_classes(wanted_only=True))
  classes = get_classes(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  int2label = get_int2label(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  model_settings = prepare_model_settings(
      label_count=len(prepare_words_list(classes)), sample_rate=sample_rate,
      clip_duration_ms=1000, window_size_ms=25.0, window_stride_ms=15.0,
      dct_coefficient_count=80, num_log_mel_features=60,
      output_representation=output_representation)
  ap = AudioProcessor(
      data_dirs=['data/train/audio'], wanted_words=classes,
      silence_percentage=12.0, unknown_percentage=5.0,
      validation_percentage=10.0, testing_percentage=0.0,
      model_settings=model_settings,
      output_representation=output_representation)
  model = load_model('checkpoints_177/ep-070-vl-0.2478.hdf5',
                     custom_objects={'relu6': relu6,
                                     'DepthwiseConv2D': DepthwiseConv2D,
                                     'overlapping_time_slice_stack':
                                     overlapping_time_slice_stack,
                                     'softmax': softmax,
                                     '<lambda>':
                                     smooth_categorical_crossentropy})
  # embed()

  # In wanted_labels we map the not wanted words to `unknown`. Though we
  # keep track of all labels in `labels`.
  fns, wanted_labels, labels, probabilities = [], [], [], []
  batch_counter = 0
  X_batch = []
  X_tta_batch = []
  if output_representation == 'mfcc_and_raw':
    X_batch = [[], []]
    X_tta_batch = [[], []]

  for i in tqdm(range(len(test_fns[:]))):
    test_fn = test_fns[i]
    fns.append(os.path.basename(test_fn))
    feed_dict = {
        ap.wav_filename_placeholder_: test_fn,
        ap.background_volume_placeholder_: 0.0,
        ap.background_data_placeholder_: np.zeros(
            (model_settings['desired_samples'], 1)),
        ap.foreground_volume_placeholder_: 1.0,
        ap.time_shift_placeholder_: 0,
    }
    if output_representation == 'raw':
      raw_val = sess.run(
          ap.background_clamp_, feed_dict=feed_dict).flatten()
      X_batch.append(raw_val)
      if use_speed_tta:
        raw_val = sess.run(
            ap.background_clamp_, feed_dict=feed_dict).flatten()
        X_tta_batch.append(raw_val)
    elif output_representation == 'spec':
      spec_val = sess.run(
          ap.spectrogram_, feed_dict=feed_dict).flatten()
      X_batch.append(spec_val)
    elif output_representation == 'mfcc':
      mfcc_val = sess.run(
          ap.mfcc_, feed_dict=feed_dict).flatten()
      X_batch.apped(mfcc_val)
    elif output_representation == 'mfcc_and_raw':
      raw_val, mfcc_val = sess.run(
          [ap.background_clamp_,
           ap.mfcc_], feed_dict=feed_dict)
      X_batch[0].append(mfcc_val.flatten())
      X_batch[1].append(raw_val.flatten())

    batch_counter += 1
    if batch_counter == batch_size:
      if output_representation != 'mfcc_and_raw':
        probs = model.predict(np.float32(X_batch))
      else:
        X_arr = [np.float32(X_batch[0]), np.float32(X_batch[1])]
        probs = model.predict(X_arr)

      if use_tta:
        X_batch_left = np.roll(np.float32(X_batch), -500, axis=1)
        left_probs = model.predict(X_batch_left)

        loud_probs = model.predict(
            np.clip(1.1 * np.float32(X_batch), -1.0, 1.0))

        silent_probs = model.predict(0.9 * np.float32(X_batch))
        flipped_probs = model.predict(-1.0 * np.float32(X_batch))
        if use_speed_tta:
          slow_probs = model.predict(np.float32(X_tta_batch))
          slow_loud_probs = model.predict(
              np.clip(1.1 * np.float32(X_tta_batch), -1.0, 1.0))
          slow_silent_probs = model.predict(0.9 * np.float32(X_tta_batch))

          probs = (probs +
                   loud_probs + silent_probs +
                   left_probs +
                   slow_probs + slow_loud_probs + slow_silent_probs) / 10
        else:
          probs = (probs + flipped_probs +
                   loud_probs + silent_probs +
                   left_probs) / 5

      pred = probs.argmax(axis=-1)
      probabilities.append(probs)
      pred_labels = [int2label[int(p)] for p in pred]
      pred_labels = map_to_valid(pred_labels)
      labels.extend(pred_labels)

      pred_labels = map_to_wanted(pred_labels, wanted_words)
      wanted_labels.extend(pred_labels)
      # set back counter
      batch_counter, X_batch, X_tta_batch = 0, [], []
      if output_representation == 'mfcc_and_raw':
        X_batch = [[], []]
        X_tta_batch = [[], []]

  # process remaining
  if X_batch:
    if output_representation != 'mfcc_and_raw':
      probs = model.predict(np.float32(X_batch))
    else:
      X_arr = [np.float32(X_batch[0]), np.float32(X_batch[1])]
      probs = model.predict(X_arr)

    if use_tta:
      X_batch_left = np.roll(np.float32(X_batch), -500, axis=1)
      left_probs = model.predict(X_batch_left)

      loud_probs = model.predict(
          np.clip(1.1 * np.float32(X_batch), -1.0, 1.0))

      silent_probs = model.predict(0.9 * np.float32(X_batch))
      flipped_probs = model.predict(-1.0 * np.float32(X_batch))
      if use_speed_tta:
        slow_probs = model.predict(np.float32(X_tta_batch))
        slow_loud_probs = model.predict(
            np.clip(1.1 * np.float32(X_tta_batch), -1.0, 1.0))
        slow_silent_probs = model.predict(0.9 * np.float32(X_tta_batch))

        probs = (probs +
                 loud_probs + silent_probs +
                 left_probs +
                 slow_probs + slow_loud_probs + slow_silent_probs) / 10
      else:
        probs = (probs + flipped_probs +
                 loud_probs + silent_probs +
                 left_probs) / 5

    pred = probs.argmax(axis=-1)
    probabilities.append(probs)
    pred_labels = [int2label[int(p)] for p in pred]
    pred_labels = map_to_valid(pred_labels)
    labels.extend(pred_labels)

    pred_labels = map_to_wanted(pred_labels, wanted_words)
    wanted_labels.extend(pred_labels)

  pd.DataFrame({'fname': fns, 'label': wanted_labels}).to_csv(
      'submission_177_tta_flsl.csv',
      index=False, compression=None)

  pd.DataFrame({'fname': fns, 'label': labels}).to_csv(
      'submission_177_tta_flsl_all_labels.csv',
      index=False, compression=None)

  probabilities = np.concatenate(probabilities, axis=0)
  all_data = pd.DataFrame({'fname': fns, 'label': labels})
  for i, l in int2label.items():
    all_data[l] = probabilities[:, i]
  all_data.to_csv(
      'submission_177_tta_flsl_all_labels_probs.csv',
      index=False, compression=None)
  print("Done!")
