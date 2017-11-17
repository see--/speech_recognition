import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


class AudioConverter:
  def __init__(
          self, desired_samples=16000,
          window_size_samples=480, window_stride_samples=160):
    self.wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder)
    # already pads/crops
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)
    spectrogram = contrib_audio.audio_spectrogram(
        wav_decoder.audio,
        window_size=window_size_samples,
        stride=window_stride_samples,
        magnitude_squared=True)
    self.mfcc = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=40)

  def load(self, fn, sess):
    return sess.run(
        self.mfcc,
        feed_dict={self.wav_filename_placeholder: fn})
