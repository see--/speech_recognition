from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import speech_model, prepare_model_settings
from input_data import AudioProcessor, prepare_words_list
from classes import get_classes
from IPython import embed  # noqa


def data_gen(audio_processor, sess,
             batch_size=128, background_frequency=0.8,
             background_volume_range=0.1, time_shift=(100.0 * 16000.0) / 1000,
             mode='validation'):
  while True:
    X, y = audio_processor.get_data(
        how_many=batch_size, offset=0,
        background_frequency=background_frequency,
        background_volume_range=background_volume_range,
        time_shift=time_shift, mode=mode, sess=sess)
    yield X, y


# running_mean: -0.8, running_std: 7.0
# mfcc running_mean: -0.67, running_std: 7.45
# background_clamp running_mean: -0.00064, running_std: 0.0774
# np.log(11) ~ 2.4
# np.log(12) ~ 2.5
# np.log(32) ~ 3.5
if __name__ == '__main__':
  sess = K.get_session()
  compute_mfcc = True
  sample_rate = 16000
  batch_size = 64
  classes = get_classes(wanted_only=False)
  model_settings = prepare_model_settings(
      label_count=len(prepare_words_list(classes)), sample_rate=sample_rate,
      clip_duration_ms=1000, window_size_ms=30.0, window_stride_ms=10.0,
      dct_coefficient_count=40)
  ap = AudioProcessor(
      data_dir='data/train/audio',
      silence_percentage=10.0,
      unknown_percentage=10.0,
      wanted_words=classes,
      validation_percentage=10.0,
      testing_percentage=0.0,
      model_settings=model_settings,
      compute_mfcc=compute_mfcc)
  train_gen = data_gen(ap, sess, batch_size=batch_size, mode='training')
  val_gen = data_gen(ap, sess, batch_size=batch_size, mode='validation')
  model = speech_model(
      'conv_2d_fast',
      model_settings['fingerprint_size'] if compute_mfcc else sample_rate,
      num_classes=model_settings['label_count'])
  # embed()
  model.fit_generator(
      train_gen, ap.set_size('training') // batch_size,
      epochs=40, verbose=1, callbacks=[
          ModelCheckpoint(
              'checkpoints_005/ep-{epoch:03d}-val_loss-{val_loss:.3f}.hdf5'),
          LearningRateScheduler(lambda ep: 0.005 if ep < 20 else 0.0005)],
      validation_data=val_gen,
      validation_steps=ap.set_size('validation') // batch_size)
  eval_res = model.evaluate_generator(
      val_gen, ap.set_size('validation') // batch_size)
  print(eval_res)
