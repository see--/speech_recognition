from __future__ import division, print_function
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from callbacks import ConfusionMatrixCallback
from model import speech_model, prepare_model_settings
from input_data import AudioProcessor, prepare_words_list
from classes import get_classes
from IPython import embed  # noqa


def data_gen(audio_processor, sess,
             batch_size=128,
             background_frequency=0.7, background_volume_range=0.2,
             foreground_frequency=0.7, foreground_volume_range=0.2,
             time_shift_frequency=0.7, time_shift_range=[-2000, 0],
             mode='validation', pseudo_frequency=0.05):
  offset = 0
  if mode != 'training':
    background_frequency = 0.0
    background_volume_range = 0.0
    foreground_frequency = 0.0
    foreground_volume_range = 0.0
    pseudo_frequency = 0.0
    time_shift_frequency = 0.0
    time_shift_range = [0, 0]
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
        pseudo_frequency=pseudo_frequency)
    offset += batch_size
    if offset > ap.set_size(mode) - batch_size:
      offset = 0
    yield X, y


# running_mean: -0.8 | running_std: 7.0
# mfcc running_mean: -0.67 | running_std: 7.45
# background_clamp running_mean: -0.00064 | running_std: 0.0774, p5: -0.074, p95: 0.0697  # noqa
# 10 ** raw - 1.0 running_mean: 0.017 | 10 ** raw - 1.0 running_std: 0.28
# np.log(11) ~ 2.4
# np.log(12) ~ 2.5
# np.log(32) ~ 3.5
# np.log(48) ~ 3.9
# 64727 training files
if __name__ == '__main__':
  # restrict gpu usage: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory  # noqa
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  K.set_session(sess)
  data_dirs = ['data/train/audio']
  add_pseudo = True
  if add_pseudo:
    data_dirs.append('data/pseudo/audio')
  output_representation = 'mfcc'
  sample_rate = 16000
  batch_size = 512
  classes = get_classes(wanted_only=False, extend_reversed=False)
  model_settings = prepare_model_settings(
      label_count=len(prepare_words_list(classes)), sample_rate=sample_rate,
      clip_duration_ms=1000, window_size_ms=30.0, window_stride_ms=15.0,
      dct_coefficient_count=60)
  ap = AudioProcessor(
      data_dirs=data_dirs, wanted_words=classes,
      silence_percentage=15.0, unknown_percentage=5.0,
      validation_percentage=10.0, testing_percentage=0.0,
      model_settings=model_settings,
      output_representation=output_representation)
  train_gen = data_gen(ap, sess, batch_size=batch_size, mode='training')
  val_gen = data_gen(ap, sess, batch_size=batch_size, mode='validation')
  model = speech_model(
      'conv_1d_log_mfcc',
      model_settings['fingerprint_size'] if output_representation == 'mfcc' else sample_rate,  # noqa
      num_classes=model_settings['label_count'])
  # embed()
  callbacks = [
      ConfusionMatrixCallback(
          val_gen, ap.set_size('validation') // batch_size,
          wanted_words=prepare_words_list(get_classes(wanted_only=True)),
          all_words=prepare_words_list(classes),
          label2int=ap.word_to_index),
      ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max',
                        factor=0.5, patience=4, verbose=1),
      TensorBoard(log_dir='logs_118'),
      ModelCheckpoint('checkpoints_118/ep-{epoch:03d}-vl-{val_loss:.4f}.hdf5')]
  model.fit_generator(
      train_gen, steps_per_epoch=ap.set_size('training') // batch_size,
      epochs=200, verbose=1, callbacks=callbacks)

  eval_res = model.evaluate_generator(
      val_gen, ap.set_size('validation') // batch_size)
  print(eval_res)
