import numpy as np
from pandas_ml import ConfusionMatrix
from keras.callbacks import Callback


class ConfusionMatrixCallback(Callback):
  def __init__(self, validation_data, validation_steps,
               wanted_words, all_words, label2int):
    self.validation_data = validation_data
    self.validation_steps = validation_steps
    self.wanted_words = wanted_words
    self.all_words = all_words
    self.label2int = label2int
    self.int2label = {v: k for k, v in label2int.items()}
    with open('confusion_matrix.txt', 'w'):
      pass
    with open('wanted_confusion_matrix.txt', 'w'):
      pass

  def accuracies(self, confusion_val):
    accuracies = []
    for i in range(confusion_val.shape[0]):
      num = confusion_val[i, :].sum()
      if num:
        accuracies.append(confusion_val[i, i] / num)
      else:
        accuracies.append(0.0)
    accuracies = np.float32(accuracies)
    mean_acc = accuracies.mean()
    return mean_acc

  def on_epoch_end(self, epoch, logs=None):
    y_true, y_pred = [], []
    for i in range(min(2, self.validation_steps)):
      X_batch, y_true_batch = next(self.validation_data)
      y_pred_batch = self.model.predict(X_batch)
      y_true_batch = list(y_true_batch.argmax(axis=-1))
      y_pred_batch = list(y_pred_batch.argmax(axis=-1))
      y_true_batch = [self.int2label[y] for y in y_true_batch]
      y_pred_batch = [self.int2label[y] for y in y_pred_batch]
      y_true.extend(y_true_batch)
      y_pred.extend(y_pred_batch)

    confusion = ConfusionMatrix(y_true, y_pred)
    accs = self.accuracies(confusion._df_confusion.values)
    # same for wanted words
    y_true = [y if y in self.wanted_words else '_unknown_' for y in y_true]
    y_pred = [y if y in self.wanted_words else '_unknown_' for y in y_pred]
    wanted_words_confusion = ConfusionMatrix(y_true, y_pred)
    wanted_accs = self.accuracies(
        wanted_words_confusion._df_confusion.values)
    print("\n[%03d]: mAcc (all): %.2f, mAcc (wanted): %.2f" %
          (epoch, accs.mean(), wanted_accs.mean()))
    # from IPython import embed; embed()
    with open('confusion_matrix.txt', 'a') as f:
      f.write('\nEpoch: %03d\n' % epoch)
      f.write(confusion.to_dataframe().to_string())

    with open('wanted_confusion_matrix.txt', 'a') as f:
      f.write('\nEpoch: %03d\n' % epoch)
      f.write(wanted_words_confusion.to_dataframe().to_string())
