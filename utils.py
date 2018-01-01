import tensorflow as tf


def data_gen(audio_processor, sess,
             batch_size=128,
             background_frequency=0.5, background_volume_range=0.2,
             foreground_frequency=0.5, foreground_volume_range=0.2,
             time_shift_frequency=0.5, time_shift_range=[-2000, 0],
             mode='validation', pseudo_frequency=0.4, flip_frequency=0.5):
  offset = 0
  if mode != 'training':
    background_frequency = 0.0
    background_volume_range = 0.0
    foreground_frequency = 0.0
    foreground_volume_range = 0.0
    pseudo_frequency = 0.0
    time_shift_frequency = 0.0
    time_shift_range = [0, 0]
    flip_frequency = 0.0
  while True:
    X, y = audio_processor.get_data(
        how_many=batch_size, offset=0 if mode == 'training' else offset,
        background_frequency=background_frequency,
        background_volume_range=background_volume_range,
        foreground_frequency=foreground_frequency,
        foreground_volume_range=foreground_volume_range,
        time_shift_frequency=time_shift_frequency,
        time_shift_range=time_shift_range,
        mode=mode, sess=sess,
        pseudo_frequency=pseudo_frequency,
        flip_frequency=flip_frequency)
    offset += batch_size
    if offset > audio_processor.set_size(mode) - batch_size:
      offset = 0
    yield X, y


def tf_roll(a, shift, a_len=16000):
  # https://stackoverflow.com/questions/42651714/vector-shift-roll-in-tensorflow
  def roll_left(a, shift, a_len):
    shift %= a_len
    rolled = tf.concat(
        [a[a_len - shift:, :], a[:a_len - shift, :]], axis=0)
    return rolled

  def roll_right(a, shift, a_len):
    shift = -shift
    shift %= a_len
    rolled = tf.concat([a[shift:, :], a[:shift, :]], axis=0)
    return rolled
  # https://stackoverflow.com/questions/35833011/how-to-add-if-condition-in-a-tensorflow-graph
  return tf.cond(
      tf.greater_equal(shift, 0),
      true_fn=lambda: roll_left(a, shift, a_len),
      false_fn=lambda: roll_right(a, shift, a_len))


def center_crop(data, desired_size=16000):
    if data.ndim == 1:
      left = (len(data) - desired_size) // 2
      return data[left: left + desired_size]
    elif data.ndim == 2:
      left = (data.shape[1] - desired_size) // 2
      return data[:, left: left + desired_size]
    else:
      raise RuntimeError("Invalid tensor shape: %s" % (list(data.shape)))
