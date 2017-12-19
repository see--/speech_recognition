from librosa import effects
from tqdm import tqdm
from glob import glob
import numpy as np
from scipy.io import wavfile as wf
from os.path import join as jp, basename as bn


def main():
  tta_speed = 0.9  # slow down (i.e. < 1.0)
  sample_per_sec = 16000
  test_fns = sorted(glob('data/test/audio/*.wav'))
  tta_dir = 'data/tta_test/audio'
  for fn in tqdm(test_fns):
    basename = bn(fn)
    rate, data = wf.read(fn)
    # assert len(data) == sample_per_sec
    data = np.float32(data) / 32767
    data = effects.time_stretch(data, tta_speed)
    data = data[-sample_per_sec:]
    out_fn = jp(tta_dir, basename)
    wf.write(out_fn, rate, np.int16(data * 32767))


if __name__ == '__main__':
  main()
