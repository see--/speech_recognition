
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