from collections import OrderedDict
from input_data import prepare_words_list


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
