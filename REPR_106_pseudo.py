import pandas as pd
from tqdm import tqdm
import os
from os.path import join as jp
from shutil import copy


sub1 = pd.read_csv('submission_098_leftloud_tta_all_labels.csv')  # 87% PLB
sub2 = pd.read_csv('submission_096_leftloud_tta_all_labels.csv')  # 87% PLB
sub3 = pd.read_csv('submission_091_leftloud_tta_all_labels.csv')  # 88% PLB

consistend = ((sub1.label == sub2.label) & (sub1.label == sub3.label))
print("All: ", sub1.shape[0], " consistend: ", consistend.sum())

for i in tqdm(range(sub1.shape[0])):
  fn = sub1.loc[i, 'fname']
  if fn != sub2.loc[i, 'fname'] or fn != sub3.loc[i, 'fname']:
    print("Fatal error")
    break

  if consistend[i]:
      label = sub1.loc[i, 'label']
      dst_dir = jp('data', 'pseudo', 'audio', label)
      if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
      dst_fn = jp(dst_dir, fn)
      src_fn = jp('data', 'test', 'audio', fn)
      copy(src_fn, dst_fn, follow_symlinks=False)
