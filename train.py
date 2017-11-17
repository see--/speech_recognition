from scipy.io import wavfile as wf
from keras import backend as K
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from glob import glob
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from model import speech_model
from audio import AudioConverter
from input_data import get_label_name, data_gen
from settings import get_settings
from IPython import embed


def sanity_check(sampling_rate):
  if sampling_rate != 16000:
    print("Error invalid sampling rate: %s" % sampling_rate)
    sys.exit(0)


def border_pad(arr, padded_size=16000):
  if len(arr) == padded_size:
    return arr
  missing = padded_size - len(arr)
  pad_left = missing // 2
  pad_right = missing - pad_left
  padded_arr = np.pad(arr, (pad_left, pad_right), 'constant')
  return padded_arr


def remove_background_fns(fns, background_label='_background_noise_'):
  real_fns, background_fns = [], []
  for fn in fns:
    if background_label in fn:
      background_fns.append(fn)
    else:
      real_fns.append(fn)
  return real_fns, background_fns


def class_counts(fns, label2int_mapping):
  counts = {k: 0 for k, v in label2int_mapping.items()}
  for fn in fns:
    label_name = get_label_name(fn)
    if label_name not in counts:
      counts['unknown'] += 1
    else:
      counts[label_name] += 1
  return counts


def balance_classes(fns, label2int_mapping):
  unknown_fns = [
      fn for fn in fns if get_label_name(fn) not in label2int_mapping]
  counts = class_counts(fns, label2int_mapping)
  print(counts)
  known_counts = [v for k, v in counts.items() if k != 'unknown']
  unknown_fns = np.random.choice(
      unknown_fns, max(known_counts), replace=False)

  balanced_fns = []
  for fn in fns:
    label_name = get_label_name(fn)
    if label_name not in counts:
      if fn in unknown_fns:
        balanced_fns.append(fn)
    else:
      balanced_fns.append(fn)
  return balanced_fns


# running_mean: -0.8, running_std: 7.0
# np.log(11) ~ 2.4
if __name__ == '__main__':
  seed = 42
  sess = K.get_session()
  ac = AudioConverter()
  np.random.seed(seed)
  settings = get_settings
  batch_size = 64
  fingerprint_size = 3920
  label2int_mapping = {
      'yes': 0, 'no': 1, 'up': 2, 'down': 3,
      'left': 4, 'right': 5, 'on': 6, 'off': 7,
      'stop': 8, 'go': 9,
      'unknown': 10, 'silence': 11
  }
  int2label_mapping = {v: k for k, v in label2int_mapping.items()}
  fns = sorted(glob('data/train/audio/*/*.wav'))
  fns, background_fns = remove_background_fns(fns)
  fns = balance_classes(fns, label2int_mapping)
  train_fns, val_fns = train_test_split(
      fns, test_size=0.2, random_state=seed)
  train_gen = data_gen(train_fns, batch_size, label2int_mapping, ac, sess)
  val_gen = data_gen(val_fns, batch_size, label2int_mapping, ac, sess)
  model = speech_model('snn', fingerprint_size)
  model.load_weights('final.hdf5')
  model.fit_generator(
      train_gen, len(train_fns) // batch_size,
      epochs=20, verbose=1, callbacks=[],
      validation_data=val_gen, validation_steps=len(val_fns) // batch_size)
  model.save_weights('final_002.hdf5')
  eval_res = model.evaluate_generator(val_gen, len(val_fns) // batch_size)
  print(eval_res)
