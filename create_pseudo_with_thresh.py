import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd


AUDIO_NAMES = ['silence', 'unknown', 'yes', 'no', 'up', 'down',
               'left', 'right', 'on', 'off', 'stop', 'go']
num_small_prob = 0
num_labels = 0
# get filenames in correct order
fnames = pd.read_csv('submission_50.csv').fname.values
probs = np.memmap('submit_50_probs.uint8.memmap',
                  mode='r+', shape=(158538, 12))
max_probs = np.float32(probs.max(axis=-1)) / 255
preds = probs.argmax(axis=-1)
prob_thresh = 0.6
pseudo_dir = 'data/heng_pseudo'
silence_count = 0
if not os.path.exists(pseudo_dir):
  os.makedirs(pseudo_dir)
else:
  shutil.rmtree(pseudo_dir)
  os.makedirs(pseudo_dir)

for i in tqdm(range(len(fnames))):
  fn = fnames[i]
  label = AUDIO_NAMES[preds[i]]
  dir_name = os.path.join(pseudo_dir, label)
  if not os.path.exists(dir_name):
      os.makedirs(dir_name)

  p = max_probs[i]
  if p < prob_thresh:
      num_small_prob += 1
      if label == 'silence':
          silence_count += 1
      continue

  if label == 'silence':
      silence_count += 1
      continue

  src_fn = os.path.join('data/test/audio', fn)
  dst_fn = os.path.join(pseudo_dir, label, fn)
  shutil.copy(src_fn, dst_fn)
  num_labels += 1

print("%d of %d pseudo labels were created." % (num_labels, len(preds)))
print("%d of %d have low prob" % (num_small_prob, len(preds)))
