from acoustics.generator import noise
from scipy.io import wavfile as wf
from os.path import join as jp
import numpy as np


if __name__ == '__main__':
  noise_dir = jp('data', 'train', 'audio', '_background_noise_')
  white_noise_fn = jp(noise_dir, 'custom_white_noise.wav')
  # 'white' and 'pink' already exist
  noise_colors = ['blue', 'brown', 'violet']
  for noise_color in noise_colors:
    noise_fn = jp(noise_dir, 'custom_%s_noise.wav' % noise_color)
    wf.write(
        noise_fn, 16000,
        np.int16(((noise(16000 * 60, color=noise_color)) / 3) * 32767))
  print("Done!")
