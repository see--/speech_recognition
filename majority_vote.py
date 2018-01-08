import pandas as pd
from os.path import join as jp
from shutil import copy
from tqdm import tqdm
from IPython import embed


# sub_fns = [
#     'submission_011.csv', 'submission_017.csv', 'submission_018.csv',
#     'submission_026.csv', 'submission_020.csv', 'submission_031.csv',
#     'submission_022.csv', 'submission_033b.csv', 'submission_027.csv',
#     'submission_044.csv', 'submission_043.csv', 'submission_052.csv',
#     'submission_051.csv', 'submission_059.csv']

sub_fns = [
    'submission_106_tta_leftloud.csv',  # 88% <-- best
    'submission_112_tta_silentloudleftleft.csv',  # 88%
    'submission_173_tta_flsl.csv',  # 88%
    'submission_143_tta_sllll.csv',  # 88%
    'submission_091_leftsilentloud_tta.csv']  # 87%
subs = [pd.read_csv(sub_fn) for sub_fn in sub_fns]

min_count = 3
fname, label, voted_count = [], [], []
clear_majority = 0
for i in tqdm(range(subs[0].shape[0])):
  fname.append(subs[0].loc[i, 'fname'])
  label_counts = {}
  for sub in subs:
    ll = sub.loc[i, 'label']
    if ll in label_counts:
      label_counts[ll] += 1
    else:
      label_counts[ll] = 1
  maj_label = max(label_counts, key=label_counts.get)
  maj_count = max(label_counts.values())
  if maj_count >= min_count:
    clear_majority += 1
  else:
    # in trouble save the wav files!
    src = jp('data', 'test', 'audio', fname[-1])
    dst_bn = str(label_counts).strip('{}').replace(" ", "").replace("'", "")  \
        + '_' + fname[-1]
    dst_bn = dst_bn.replace(":", "_").replace(",", "_")
    dst = jp('split_decision', dst_bn)
    copy(src, dst)
    # a.) resolve tie by chosing the best submission based on the PLB score
    maj_label = subs[0].loc[i, 'label']

    # b.) resolve tie by chosing unknown before silence
    # if label_counts.get('silence', 0) == maj_count:
    #   maj_label = 'silence'
    # if label_counts.get('unknown', 0) == maj_count:
    #   maj_label = 'unknown'
  label.append(maj_label)
  voted_count.append(min_count)


pd.DataFrame({'fname': fname, 'label': label}).to_csv(
    'majority_sub_034.csv', index=False)
print("Done! Got a clear majority for %d of %d samples."
      % (clear_majority, subs[0].shape[0]))

pd.DataFrame({'fname': fname, 'label': label, 'count': voted_count}).to_csv(
    'majority_sub_034_count.csv', index=False)
