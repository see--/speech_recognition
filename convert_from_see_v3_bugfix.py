import pandas as pd
import numpy as np
from collections import OrderedDict


SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'


def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def get_classes(wanted_only=False, extend_reversed=False):
  if wanted_only:
    classes = 'stop down off right up go on yes left no'
    classes = classes.split(' ')
    assert len(classes) == 10
  else:
    classes =  'sheila nine stop bed four six down bird marvin cat off right seven eight up three happy go zero on wow dog yes five one tree house two left no' # noqa
    classes = classes.split(' ')
    assert len(classes) == 30
  if extend_reversed:
    assert not wanted_only
    new_classes = ['new_owt', 'new_yppah', 'new_xis', 'new_esuoh',
                   'new_neves', 'new_thgie', 'new_ruof', 'new_tac',
                   'new_nivram', 'new_enin', 'new_aliehs', 'new_eert',
                   'new_orez', 'new_eerht', 'new_evif', 'new_deb',
                   'new_drib']
    assert len(new_classes) == 17
    classes.extend(new_classes)
  return classes


def get_int2label(wanted_only=False, extend_reversed=False):
  classes = get_classes(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  classes = prepare_words_list(classes)
  int2label = {i: l for i, l in enumerate(classes)}
  int2label = OrderedDict(sorted(int2label.items(), key=lambda x: x[0]))
  return int2label


def get_label2int(wanted_only=False, extend_reversed=False):
  classes = get_classes(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  classes = prepare_words_list(classes)
  label2int = {l: i for i, l in enumerate(classes)}
  label2int = OrderedDict(sorted(label2int.items(), key=lambda x: x[1]))
  return label2int


def softmax(x):
  exp_prob = np.exp(x)
  return exp_prob / exp_prob.sum(axis=1, keepdims=True)


NUM_AUDIO_TEST_SAMPLES = 158538
AUDIO_NAMES = ['silence', 'unknown', 'yes', 'no', 'up', 'down',
               'left', 'right', 'on', 'off', 'stop', 'go']
AUDIO_NUM_CLASSES = len(AUDIO_NAMES)
int2label = get_int2label(wanted_only=False)
label2int = get_label2int(wanted_only=False)

see_file = '../submission_106_tta_leftloud_all_labels_probs.csv'
memmap_file = 'submission_106_tta_leftloud_all_labels_probs.uint8.memmap'

see_df = pd.read_csv(see_file)
all_probs = see_df.loc[:, int2label.values()].values
SEE_TEST_SAMPLES = see_df['fname'].values
SEE_MAP1 = dict(zip(SEE_TEST_SAMPLES, range(NUM_AUDIO_TEST_SAMPLES)))
see_probs = np.zeros((NUM_AUDIO_TEST_SAMPLES, AUDIO_NUM_CLASSES), np.float32)
unknown_probs = []
for i, audio_name in int2label.items():
  if audio_name == SILENCE_LABEL:
    continue

  if audio_name in AUDIO_NAMES:
    heng_idx = AUDIO_NAMES.index(audio_name)
    # print(heng_idx, AUDIO_NAMES[heng_idx], int2label[i], i)
    see_probs[:, heng_idx] = all_probs[:, i]

  else:
    print('Unknown: ', audio_name)
    unknown_probs.append(all_probs[:, i])

# silence
see_probs[:, 0] = all_probs[:, 0]

# unknown
see_probs[:, 1] = np.float32(unknown_probs).max(axis=0)
see_probs = softmax(see_probs)
print(see_probs.sum(axis=1)[:10])

# map to correct order
# MISSING

# save
norm_probs = np.memmap(
    memmap_file, dtype='uint8', mode='w+',
    shape=(NUM_AUDIO_TEST_SAMPLES, AUDIO_NUM_CLASSES))
norm_probs[...] = (see_probs * 255)
