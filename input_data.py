# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from utils import tf_roll

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 13


def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends
  on the name and the set proportions, so it won't change as other files are
  added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', 'testing' or 'pseudo'.
  """
  dir_name = os.path.basename(os.path.dirname(filename))
  if dir_name == 'unknown_unknown':
    return 'training'

  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  # TODO(see--): handle pseudo labels better
  if base_name.find('_nohash_') == -1:
    return 'pseudo'
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_dirs, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings, output_representation=False):
    self.data_dirs = data_dirs
    assert output_representation in {'raw', 'spec', 'mfcc', 'mfcc_and_raw'}
    self.output_representation = output_representation
    self.model_settings = model_settings
    for data_dir in self.data_dirs:
      self.maybe_download_and_extract_dataset(data_dir)
    self.prepare_data_index(silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage)
    self.prepare_background_data()
    self.prepare_processing_graph(model_settings)

  def maybe_download_and_extract_dataset(self, data_dir):
    if not os.path.exists(data_dir):
      print("Please download the dataset!")
      sys.exit(0)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': [],
                       'pseudo': []}
    unknown_index = {'validation': [], 'testing': [], 'training': [],
                     'pseudo': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    for data_dir in self.data_dirs:
      search_path = os.path.join(data_dir, '*', '*.wav')
      for wav_path in gfile.Glob(search_path):
        word = re.search('.*/([^/]+)/.*.wav', wav_path).group(1).lower()
        # Treat the '_background_noise_' folder as a special case,
        # since we expect it to contain long audio samples we mix in
        # to improve training.
        if word == BACKGROUND_NOISE_DIR_NAME:
          continue
        all_words[word] = True
        set_index = which_set(
            wav_path, validation_percentage, testing_percentage)
        # If it's a known class, store its detail, otherwise add it to the list
        # we'll use to train the unknown label.
        if word in wanted_words_index:
          self.data_index[set_index].append({'label': word, 'file': wav_path})
        else:
          unknown_index[set_index].append({'label': word, 'file': wav_path})
      if not all_words:
        raise Exception('No .wavs found at ' + search_path)
      for index, wanted_word in enumerate(wanted_words):
        if wanted_word not in all_words:
          raise Exception('Expected to find ' + wanted_word +
                          ' in labels but only found ' +
                          ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training', 'pseudo']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(
          unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training', 'pseudo']:
      # not really needed since the indices are chosen by random
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that
    match the sample rate of the training data, but can be much longer in
    duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dirs[0], BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dirs[0], BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_placeholder_: How much the clip is shifted.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    self.foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    self.time_shift_placeholder_ = tf.placeholder(tf.int32)
    # TODO(see--): Write test vs np.roll
    shifted_foreground = tf_roll(
        scaled_foreground, self.time_shift_placeholder_)
    # Mix in background noise.
    self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, 1])
    self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
    background_add = tf.add(background_mul, shifted_foreground)
    self.background_clamp_ = tf.clip_by_value(background_add, -1.0, 1.0)
    self.background_clamp_ = tf.reshape(
        self.background_clamp_, (1, model_settings['desired_samples']))
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    stfts = tf.contrib.signal.stft(
        self.background_clamp_,
        frame_length=model_settings['window_size_samples'],
        frame_step=model_settings['window_stride_samples'],
        fft_length=None)
    self.spectrogram_ = tf.abs(stfts)
    num_spectrogram_bins = self.spectrogram_.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
    linear_to_mel_weight_matrix = \
        tf.contrib.signal.linear_to_mel_weight_matrix(
            model_settings['dct_coefficient_count'],
            num_spectrogram_bins, model_settings['sample_rate'],
            lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        self.spectrogram_, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(self.spectrogram_.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
    self.mfcc_ = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[
        :, :, :model_settings['num_log_mel_features']]  # :13

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', 'testing',
            or 'pseudo'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset,
               background_frequency, background_volume_range,
               foreground_frequency, foreground_volume_range,
               time_shift_frequency, time_shift_range,
               mode, sess, pseudo_frequency=0.0):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be
    returned, otherwise the first N clips in the partition will be used.
    This ensures that validation always uses the same samples, reducing
    noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift_frequency: How often do we shift the
                            samples (float: 0.0 to 1.0)
      time_shift_range: How much to randomly shift the clips by in time.
                        [min_shift, max_shift]
      mode: Which partition to use, must be 'training', 'validation',
        'testing' or 'pseudo'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of labels in
      one-hot form.
    """
    # Pick one of the partitions to choose samples from.
    model_settings = self.model_settings
    candidates = self.data_index[mode]
    pseudo_candidates = self.data_index['pseudo']
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    if self.output_representation == 'raw':
      data_dim = model_settings['desired_samples']
    elif self.output_representation == 'spec':
      data_dim = model_settings['spectrogram_length'] * model_settings[
          'spectrogram_frequencies']
    elif self.output_representation == 'mfcc':
      data_dim = model_settings['spectrogram_length'] * \
          model_settings['num_log_mel_features']
    elif self.output_representation == 'mfcc_and_raw':
      data_dim = model_settings['spectrogram_length'] * \
          model_settings['num_log_mel_features']
      raw_data = np.zeros((sample_count, model_settings['desired_samples']))

    data = np.zeros((sample_count, data_dim))
    labels = np.zeros((sample_count, model_settings['label_count']))
    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
        sample = candidates[sample_index]
      else:
        if np.random.uniform(0, 1) < pseudo_frequency:
          sample_index = np.random.randint(len(pseudo_candidates))
          sample = pseudo_candidates[sample_index]
        else:
          sample_index = np.random.randint(len(candidates))
          sample = candidates[sample_index]

      # If we're time shifting, set up the offset for this sample.
      if np.random.uniform(0.0, 1.0) < time_shift_frequency:
        time_shift = np.random.randint(
            time_shift_range[0], time_shift_range[1] + 1)
      else:
        time_shift = 0
      input_dict = {
          self.wav_filename_placeholder_: sample['file'],
          self.time_shift_placeholder_: time_shift,
      }
      # Choose a section of background noise to mix in.
      if use_background:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples, 1])
        if np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0.0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0.0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        input_dict[self.foreground_volume_placeholder_] = 0.0
      else:
        # Turn it up or down
        if np.random.uniform(0, 1) < foreground_frequency:
          foreground_volume = 1.0 + np.random.uniform(
              -foreground_volume_range, foreground_volume_range)
          input_dict[self.foreground_volume_placeholder_] = foreground_volume
        else:
          input_dict[self.foreground_volume_placeholder_] = 1.0
      # Run the graph to produce the output audio.
      if self.output_representation == 'raw':
        data[i - offset, :] = sess.run(
            self.background_clamp_, feed_dict=input_dict).flatten()
      elif self.output_representation == 'spec':
        data[i - offset, :] = sess.run(
            self.spectrogram_, feed_dict=input_dict).flatten()
      elif self.output_representation == 'mfcc':
        data[i - offset, :] = sess.run(
            self.mfcc_, feed_dict=input_dict).flatten()
      elif self.output_representation == 'mfcc_and_raw':
        raw_val, mfcc_val = sess.run(
            [self.background_clamp_,
             self.mfcc_], feed_dict=input_dict)
        data[i - offset, :] = mfcc_val.flatten()
        raw_data[i - offset, :] = raw_val.flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset, label_index] = 1

    if self.output_representation != 'mfcc_and_raw':
      return data, labels
    else:
      return [data, raw_data], labels

  def get_unprocessed_data(self, how_many, model_settings, mode):
    """Retrieve sample data for the given partition, with no transformations.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation',
        'testing' or 'pseudo'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = model_settings['desired_samples']
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = []
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
        data[i, :] = sess.run(
            scaled_foreground, feed_dict=input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])
    return data, labels

  def summary(self):
    """Prints a summary of the used classes and the label distributions"""
    set_counts = {}
    print("There are %d classes." % (len(self.word_to_index)))
    print("1%% <-> %d samples in 'training'" %
          int(self.set_size('training') / 100))
    for set_index in ['training', 'validation', 'testing', 'pseudo']:
      counts = {k: 0 for k in sorted(self.word_to_index.keys())}
      num_total = self.set_size(set_index)
      for data_point in self.data_index[set_index]:
        counts[data_point['label']] += (1.0 / num_total) * 100.0
      set_counts[set_index] = counts

    print("%-13s%-6s%-6s%-6s%-6s" % ('', 'Train', 'Val', 'Test', 'Pseudo'))
    for label_name in sorted(
            self.word_to_index.keys(), key=self.word_to_index.get):
      line = "%02d %-12s: " % (self.word_to_index[label_name], label_name)
      for set_index in ['training', 'validation', 'testing', 'pseudo']:
        line += "%.1f%% " % (set_counts[set_index][label_name])
      print(line)
