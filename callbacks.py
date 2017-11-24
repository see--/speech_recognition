import numpy as np
from pandas_ml import ConfusionMatrix
from keras.callbacks import Callback


def log_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1. - eps)
    ce = -(np.sum(y_true * np.log(y_pred), axis=1))
    mce = ce.mean()
    return mce


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
    return accuracies

  def on_epoch_end(self, epoch, logs=None):
    y_true, y_pred = [], []
    for i in range(self.validation_steps):
      X_batch, y_true_batch = next(self.validation_data)
      y_pred_batch = self.model.predict(X_batch)

      y_true.extend(y_true_batch)
      y_pred.extend(y_pred_batch)

    y_true = np.float32(y_true)
    y_pred = np.float32(y_pred)
    val_loss = log_loss(y_true, y_pred)
    # map integer labels to strings
    y_true = list(y_true.argmax(axis=-1))
    y_pred = list(y_pred.argmax(axis=-1))
    y_true = [self.int2label[y] for y in y_true]
    y_pred = [self.int2label[y] for y in y_pred]
    confusion = ConfusionMatrix(y_true, y_pred)
    accs = self.accuracies(confusion._df_confusion.values)
    # same for wanted words
    y_true = [y if y in self.wanted_words else '_unknown_' for y in y_true]
    y_pred = [y if y in self.wanted_words else '_unknown_' for y in y_pred]
    wanted_words_confusion = ConfusionMatrix(y_true, y_pred)
    wanted_accs = self.accuracies(
        wanted_words_confusion._df_confusion.values)
    acc_line = "\n[%03d]: val_categorical_accuracy: %.2f, val_categorical_accuracy_wanted: %.2f" % (epoch, accs.mean(), wanted_accs.mean())  # noqa
    with open('confusion_matrix.txt', 'a') as f:
      f.write('%s\n' % acc_line)
      f.write(confusion.to_dataframe().to_string())

    with open('wanted_confusion_matrix.txt', 'a') as f:
      f.write('%s\n' % acc_line)
      f.write(wanted_words_confusion.to_dataframe().to_string())

    logs['val_loss'] = val_loss
    logs['val_categorical_accuracy'] = accs.mean()
    logs['val_categorical_accuracy_wanted'] = wanted_accs.mean()
