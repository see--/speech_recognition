from __future__ import division
import tensorflow as tf
from keras import backend as K


def data_gen(audio_processor, sess,
             batch_size=128,
             background_frequency=0.3, background_volume_range=0.15,
             foreground_frequency=0.3, foreground_volume_range=0.15,
             time_shift_frequency=0.3, time_shift_range=[-500, 0],
             mode='validation', pseudo_frequency=0.33, flip_frequency=0.0,
             silence_volume_range=0.3):
  ep_count = 0
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
    # silence_volume_range: stays the same for validation
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
        flip_frequency=flip_frequency,
        silence_volume_range=silence_volume_range)
    offset += batch_size
    if offset > audio_processor.set_size(mode) - batch_size:
      offset = 0
      # if mode == 'training':
      #   if 20 >= ep_count:
      #     pseudo_frequency = 1.0
      #   elif 30 >= ep_count > 20:
      #     pseudo_frequency = 0.7
      #   elif 40 >= ep_count > 30:
      #     pseudo_frequency = 0.4
      #   else:
      #     pseudo_frequency = 0.2
      print("\n[Ep:%03d: %s-mode]: Pseudo: %.3f"
            % (ep_count, mode, pseudo_frequency))
      ep_count += 1
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


def smooth_categorical_crossentropy(
        target, output, from_logits=False, label_smoothing=0.0):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output)

    return tf.losses.softmax_cross_entropy(
        target, output, label_smoothing=label_smoothing)
