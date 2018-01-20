# TensorFlow Speech Recognition Challenge
This repo contains my part of the code for our winning entry in the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) hosted by [kaggle](https://www.kaggle.com). The task was to *build an algorithm that understands simple speech commands*.
Our team (team name: **Heng-Ryan-See \* good bug? \***) won the competition with a private leaderboard score of: 0.91060.

## Members

- [Heng Cher Keng](https://www.kaggle.com/hengck23)
- [Ryan Sun](https://www.kaggle.com/ryansun)
- [Steffen Eberbach](https://www.kaggle.com/seesee)

# Overview of my approach
I started with the provided [tutorial](https://www.tensorflow.org/versions/master/tutorials/audio_recognition) and could easily get better results by just adding momentum to the plain SGD solver (82-83% on the leaderboard). I have no prior experience with audio data and mostly used deep learning with images. For this domain you don't use features but feed the raw pixel values. My thinking was that this should work with audio data as well. Throughout the competition I ran experiments using raw waveforms, spectrograms and log mel features as input. I got similar results using log mel and raw waveform (86%-87%) and used the waveform data for most experiments as it was easier to interpret for me.

For the special price the restrictions were: the network is smaller than 5.000.000 bytes and runs in less than 175ms per sample on a stock Raspberry Pi 3. Regarding the size, this allows you to build networks that have roughly 1.250.000 weight parameters. So by experimenting with these restrictions I came up with an architecture that uses Depthwise1D convolutions on the raw waveform. Using [model distillation](https://arxiv.org/pdf/1503.02531.pdf) this network predicts the correct class for 90.8% of the private leaderboard samples and runs in roughly 80ms.

# What didn't work

- Fancy augmentation methods: I tried flipping (i.e: ` * -1.0`) the samples. You can check that they will sound exactly the same. I also modified `input_data.py` to change the foreground and background volume independently and created a separate volume range for the silence samples. My validation accuracy improved for some experiments but my leaderboard scores didn't.

- Predicting unknown unknowns: I didn't find a good way to consistently predict these words. Often, similar words were wrongly classified (e.g. one as on).

- Creating new words: I trained some networks with even more classes. I reversed the samples from the known unwanted words, e.g. `bird`, `bed`, `marvin`, and created new classes (`bird` -> `drib` ...). The idea was to have more unknowns to prevent the network from wrongly mapping unknowns to the known words. For example the word `follow` was mostly predicted as `off`. However, neither my validation score not my leaderboard score improved.

- Cyclic learning rate schedules: The winning entry of the [Caravana Image Masking Challenge](http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/) used cyclic learning rates but for me the results got worse and you had additional hyperparameters. Maybe I just didn't implement it correctly.

# What worked
- Mixing tensorflow and Keras: Both frameworks work perfectly together and you can mix them wherever you want. For example: I wrapped the provided data AudioProcessor from `input_data.py` in a generator and used it with `keras.models.Model.fit_generator`. This way, I could implement new architectures really fast using Keras and later just extract and freeze the graph from the trained models (see `freeze_graph.py`).

- Pseudo labeling: I used consistent samples from the test set to train new networks. Choosing them was based on a.) my three best models agree on this submission. I used this version at early stages of the competition. b.) using a probability threshold on the predicted softmax probabilities. Typically, using `pseudo_threshold=0.6` were the samples that our ensembled model predicted correctly. I also implemented a schedule for pseudo labels. That is: For the first 5 epochs you only use pseudo labels and then gradually mix in data from the training data set. Though, I didn't have time to run these experiments, so I kept a fixed ratio of training and pseudo data.

- Test time augmentation: It is a simple way to get some boost. Just augment the samples, feed them multiple times and average the probabilities. I tried the following: time-shifting, increase/decrease the volume and time-stretching using `librosa.effects.time_stretch`.

# Structure
This repo contains all the code (`.py` or `.ipynb`) to reproduce my part of our submission. You'll find various model definitions in `model.py`, the training script is `train.py` and the scripts to make the submissions are `make_submission.py` (faster as it processes samples in batches) or `make_submission_on_rpi.py` (suitable to create the submission file on the rpi 3: frozen graph, batch size of 1 and fewer dependecies). Keras models checkpoints can be found in `checkpoints_*`, TensorBoard training logs in `logs_*` and frozen TensorFlow graphs in `tf_files`. I am providing these files for the experiments that are required for the final submission. Though, I ran many more. As a result each section of this writeup can be executed independently.
The data is assumed to be in `data/train` and `data/test`:
```
mkdir data
ln -s your/path/to/train data/train
ln -s your/path/to/test data/test
```

# Requirements:
- tensorflow-gpu==1.4.0 
- Keras==2.1.2 (the version that comes with tensorflow `2.0.8-tf` should be fine as well but you'll need to adjust the imports)
- tqdm==4.11.2
- scipy==1.0.0
- numpy==1.13.3
- pandas==0.20.3
- pandas-ml==0.5.0

All packages can be installed via `pip3 install`. Other versions will probably work too. I tested it with Python 3.5.2 using Ubuntu 16.04.

## Model training
Only the predicted probabilites by the model from experiment 106 made it to our final ensemble submission.
I trained a lot more but this one was the best. This model is trained with pseudo labels. So the first step is to reproduce them:
```
git checkout 6892d80
git checkout master submission_091_leftloud_tta_all_labels.csv submission_096_leftloud_tta_all_labels.csv submission_098_leftloud_tta_all_labels.csv REPR_explore.ipynb
python3 generate_noise.py
jupyter notebook REPR_explore.ipynb
```
Then run the Notebook cells that produces the pseudo labels: the first one and the 3 cells following: **# Create pseudo labels from consistent predictions
**. Later in the competition this step is replaced by the `create_pseudo_with_thresh.py` script. Close the notebook (otherwise the GPU memory is still occupied) and train the model:
```
mkdir checkpoints_106
python3 train.py
```
For the submission, I selected the checkpoint with the highest validation accuracy using tensorboard:
```
tensorboard --logdir logs_106
```
The reference model is `checkpoints_106/ep-062-vl-0.1815.hdf5`. This experiment itself requires submissions by other networks. To train these models run ....

 Note that due to stochasticity (random initialization, data augmentation and me not setting a seed) exactly reproducing these weights is probably not possible.

## Make the submission using TTA:
```
git checkout master make_submission.py
git checkout master checkpoints_106/ep-062-vl-0.1815.hdf5  # change line 64 of `make_submission.py` instead of this command if you use another checkpoint
python3 make_submission.py
```

The resulting submission will have a private/public score of 0.88558/0.88349. Every sample is used three times (unchanged, shifted to the left by 1500 timesteps and made louder by multiplying with 1.2). The resulting probabilities are then averaged. Note that this model uses 32 classes. These probabilities will be stored in `REPR_submission_106_tta_leftloud_all_labels_probs.csv`. In order to use them for the ensembled model the order of the samples and the probabilities have to be converted:
```
git checkout master convert_from_see_v3_bugfix.py
python3 convert_from_see_v3_bugfix.py
```

## Raspberry Pi model
This model is trained with pseudo labels from our best ensembled submission: `submit_50_probs.uint8.memmap`. To train this model run:
```
git checkout 4f22e26
python3 train.py
```
For this training I am only saving the checkpoints with the best validation accuracy. Therefore, there is no need to inspect the logs. Just use the latest checkpoint.

To freeze the model run:
```
git checkout master freeze_graph.py
git checkout master checkpoints_195/ep-085-vl-0.2231.hdf5  # skip this if you trained the model yourself
python3 freeze_graph.py --checkpoint_path checkpoints_195/ep-085-vl-0.2231.hdf5
```
By default, the frozen graph will be saved as `tf_files/frozen.pb`. You can then reproduce the best scoring rpi submission `rpi_submission_195.csv` by running:
```
git checkout master make_submission_on_rpi.py
python3 make_submission_on_rpi.py --frozen_graph tf_files/frozen.pb --test_data data/test/audio --submission_fn rpi_submission_195.csv
```
This submission will have a score of 0.90825 on the private leaderboard.

## Benchmarking the Pi model
You can benchmark the frozen graph using the provided benchmark binary:

```
curl -O https://storage.googleapis.com/download.tensorflow.org/deps/pi/2017_10_07/benchmark_model
chmod +x benchmark_model
./benchmark_model --graph=tf_files/frozen.pb --input_layer="decoded_sample_data:0,decoded_sample_data:1" --input_layer_shape="16000,1:" --input_layer_type="float,int32" --input_layer_values=":16000" --output_layer="labels_softmax:0" --show_run_order=false --show_time=false --show_memory=false --show_summary=true --show_flops=true```

I got the following results:
- avg time (ms): 58.042
- max memory (bytes): 2180436
- size (bytes): 4870144 (`du -s -B1 tf_files/frozen.pb`)