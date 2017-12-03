from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import load_model
from model import prepare_model_settings
from input_data import prepare_words_list
from classes import get_classes, get_int2label
from IPython import embed  # noqa


if __name__ == '__main__':
  test_fns = sorted(glob('data/test/audio/*.wav'))
  sess = K.get_session()
  wanted_only = True
  compute_mfcc = False
  sample_rate = 16000
  batch_size = 64
  wanted_words = prepare_words_list(get_classes(wanted_only=True))
  classes = get_classes(wanted_only=wanted_only)
  int2label = get_int2label(wanted_only=wanted_only)
  model_settings = prepare_model_settings(
      label_count=len(prepare_words_list(classes)), sample_rate=sample_rate,
      clip_duration_ms=1000, window_size_ms=30.0, window_stride_ms=10.0,
      dct_coefficient_count=40)

  wav_filename_placeholder = tf.placeholder(tf.string, [])
  wav_loader = io_ops.read_file(wav_filename_placeholder)
  wav_decoder = contrib_audio.decode_wav(
      wav_loader, desired_channels=1,
      desired_samples=model_settings['desired_samples'])
  clamped = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
  spectrogram = contrib_audio.audio_spectrogram(
      clamped,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)
  mfcc = contrib_audio.mfcc(
      spectrogram,
      wav_decoder.sample_rate,
      dct_coefficient_count=model_settings['dct_coefficient_count'])
  # embed()
  model = load_model('checkpoints_031/ep-053-vl-0.2193.hdf5')
  # In wanted_labels we map the not wanted words to `unknown`. Though we
  # keep track of all labels in `labels`.
  fns, wanted_labels, labels, probabilities = [], [], [], []
  batch_counter = 0
  X_batch = []
  for test_fn in tqdm(test_fns[:]):
    fns.append(os.path.basename(test_fn))
    if compute_mfcc:
      mfcc_val = sess.run(mfcc, {wav_filename_placeholder: test_fn})
      X_batch.append(mfcc_val.flatten())
    else:
      raw_val = sess.run(clamped, {wav_filename_placeholder: test_fn})
      X_batch.append(raw_val.flatten())

    batch_counter += 1
    if batch_counter == batch_size:
      pred = model.predict(np.float32(X_batch))
      probabilities.append(pred)
      pred = pred.argmax(axis=-1)
      pred_labels = [int2label[int(p)] for p in pred]
      # map '_silence_' to 'silence'
      pred_labels = [
          pl if pl != '_silence_' else 'silence' for pl in pred_labels]
      # map '_unknown_' to 'unknown'
      pred_labels = [
          pl if pl != '_unknown_' else 'unknown' for pl in pred_labels]
      labels.extend(pred_labels)

      # map unknown words to 'unknown'
      pred_labels = [
          pl if pl in wanted_words or pl == 'silence'
          else 'unknown' for pl in pred_labels]
      wanted_labels.extend(pred_labels)
      # set back counter
      batch_counter, X_batch = 0, []

  # process remaining
  if X_batch:
    pred = model.predict(np.float32(X_batch))
    probabilities.append(pred)
    pred = pred.argmax(axis=-1)
    pred_labels = [int2label[int(p)] for p in pred]
    # map '_silence_' to 'silence'
    pred_labels = [
        pl if pl != '_silence_' else 'silence' for pl in pred_labels]
    # map '_unknown_' to 'unknown'
    pred_labels = [
        pl if pl != '_unknown_' else 'unknown' for pl in pred_labels]
    labels.extend(pred_labels)

    # map unknown words to 'unknown'
    pred_labels = [
        pl if pl in wanted_words or pl == 'silence'
        else 'unknown' for pl in pred_labels]
    wanted_labels.extend(pred_labels)

  pd.DataFrame({'fname': fns, 'label': wanted_labels}).to_csv(
      'submission_031.csv', index=False, compression=None)

  pd.DataFrame({'fname': fns, 'label': labels}).to_csv(
      'submission_031_all_labels.csv', index=False, compression=None)

  probabilities = np.concatenate(probabilities, axis=0)
  all_data = pd.DataFrame({'fname': fns, 'label': labels})
  for i, l in int2label.items():
    all_data[l] = probabilities[:, i]
  all_data.to_csv(
      'submission_031_all_labels_probs.csv', index=False, compression=None)
  print("Done!")
