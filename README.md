# TensorFlow Speech Recognition Challenge
This repo contains my part of the code for our winning entry in the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) hosted by [kaggle](https://www.kaggle.com). The task was to *build an algorithm that understands simple speech commands*.
Our team (team name: **Heng-Ryan-See \* good bug? \***) won the competition with a private leaderboard score of: 0.91060.

## Members

- [Heng Cher Keng](https://www.kaggle.com/hengck23)
- [Ryan Sun](https://www.kaggle.com/ryansun)
- [Steffen Eberbach](https://www.kaggle.com/seesee)

# Contact
- Name: Steffen Eberbach
- Location: Germany
- Email: steffenebe@gmail.com

# 1.) Background
- What your academic/professional background?
  - I studied at the Karlsruhe Institute of Technology with a focus on robotics & computer vision and prepared my master's thesis at Daimler's image understanding group.
- Did you have any prior experience that helped you succeed in this competition?
  - I already used deep learning for my master's thesis. Knowing the commonly used terms and how to interpret the learning curves helped a lot.
- What made you decide to enter this competition?
  - It was a nice way to learn something new. It was not specifically speech recognition but I wanted to use TensorFlow/Keras more anyway and the competition seemed like a good project to apply both frameworks.
- How much time did you spend on the competition?
  - Way too much :P. I needed some time to get started with audio (i.e. understanding mel features and audio representations). Then I spent (too much) time developing new network architectures. Finally, the last week and writeup took really long as well.
- If part of a team, how did you decide to team up?
  - I was somewhat stuck at 88% public leaderboard score and decided to team up. I saw that Heng did some great progression on the leaderboard and wrote him a message.
- If you competed as part of a team, who did what?
  - We discussed most problems in the group but everyone had a focus. Heng did all the ensembling. Ryan retrained his models with pseudo labeling and my focus was on the special price.


# 2.) Summary of my approach
I started with the provided [tutorial](https://www.tensorflow.org/versions/master/tutorials/audio_recognition) and could easily get better results by just adding momentum to the plain SGD solver. I have no prior experience with audio data and mostly used deep learning with images. For this domain you don't use features but feed the raw pixel values. My thinking was that this should also work with audio data. During the competition, I ran experiments using raw waveforms, spectrograms and log mel features as input. I got similar results using log mel and raw waveform (86%-87%) and used the waveform data for most experiments as it was easier to interpret.

For the special price the restrictions were: the network is smaller than 5.000.000 bytes and runs in less than 175ms per sample on a stock Raspberry Pi 3. Regarding the size, this allows you to build networks that have roughly 1.250.000 weight parameters. So by experimenting with these restrictions I came up with an architecture that uses Depthwise1D convolutions on the raw waveform. Using [model distillation](https://arxiv.org/pdf/1503.02531.pdf) this network predicts the correct class for 90.8% of the private leaderboard samples and runs in roughly 60ms. Training the model takes ~4 hours using a Google Cloud instance with K80 GPU.

## What didn't work

- Fancy augmentation methods: I tried flipping (i.e: ` * -1.0`) the samples. You can check that they will sound exactly the same. I also modified `input_data.py` to change the foreground and background volume independently and created a separate volume range for the silence samples. My validation accuracy improved for some experiments but my leaderboard scores didn't.

- Predicting unknown unknowns: I didn't find a good way to consistently predict these words. Often, similar words were wrongly classified (e.g. one as on).

- Creating new words: I trained some networks with even more classes. I reversed the samples from the known unwanted words, e.g. `bird`, `bed`, `marvin`, and created new classes (`bird` -> `drib` ...). The idea was to have more unknowns to prevent the network from wrongly mapping unknowns to the known words. For example the word `follow` was mostly predicted as `off`. However, neither my validation score not my leaderboard score improved.

- Cyclic learning rate schedules: The winning entry of the [Caravana Image Masking Challenge](http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/) used cyclic learning rates but for me the results got worse and you had additional hyperparameters. Maybe I just didn't implement it correctly.

## What worked
- Mixing tensorflow and Keras: Both frameworks work perfectly together and you can mix them wherever you want. For example: I wrapped the provided data AudioProcessor from `input_data.py` in a generator and used it with `keras.models.Model.fit_generator`. This way, I could implement new architectures really fast using Keras and later just extract and freeze the graph from the trained models (see `freeze_graph.py`).

- Pseudo labeling: I used consistent samples from the test set to train new networks. Choosing them was based on a.) my three best models agree on this submission. I used this version at early stages of the competition. b.) using a probability threshold on the predicted softmax probabilities. Typically, using `pseudo_threshold=0.6` were the samples that our ensembled model predicted correctly. I also implemented a schedule for pseudo labels. That is: For the first 5 epochs you only use pseudo labels and then gradually mix in data from the training data set. Though, I didn't have time to run these experiments, so I kept a fixed ratio of training and pseudo data.

- Test time augmentation (TTA): It is a simple way to get some boost. Just augment the samples, feed them multiple times and average the probabilities. I tried the following: time-shifting, increase/decrease the volume and time-stretching using `librosa.effects.time_stretch`.

# 3.) Feature Selection
I experimented with log mel features but eventually just used the raw waveform data.

# 4.) Training Method
- What training methods did you use?
  - I used SGD and its variants: Adam, Nadam and RMSprop. RMSprop worked best for me.
- Did you ensemble the models?
  - I think this was the most important part. Our single models were all in the range of 86%-88% accuracy on the public leaderboard. Using an ensemble of these models got us to 90%.
- If you did ensemble, how did you weight the different models?
  - Mostly based on the single model performance and by leaderboard score. We even had a typo (wrong filename) which caused one model to have twice the weight it should have. We used this submission as our final (winning) submission. Actually, Ryan proposed better ensemble weights that yielded 0.91107 on the private leaderboard.

# 5.) Interesting findings
- What do you think set you apart from others in the competition?
  - Ensembling the models and the heavy use of pseudo labeling. Using this approach, we could incrementally get a better model.

# Appendix
##  A1.) Model Execution Time
- What software did you use for training and prediction?
  - see Requirements
- What hardware (CPUS spec, number of CPU cores, memory)?
  - a single GCP instance with 4 vCPUs, 18.5 GB memory and 
1 x NVIDIA Tesla K80
- How long does it take to train your model?
  - ~4-8 hours (depends on the model) for a single model
- How long does it take to generate predictions using your model?
  - ~4 minutes without TTA, on the Pi 3 (`make_submission_on_rpi.py`) it takes ~134 minutes

## A2.) Requirements:
- tensorflow-gpu==1.4.0 
- Keras==2.1.2 (the version that comes with tensorflow `2.0.8-tf` should be fine as well but you'll need to adjust the imports)
- tqdm==4.11.2
- scipy==1.0.0
- numpy==1.13.3
- pandas==0.20.3
- pandas-ml==0.5.0

All packages can be installed via `pip3 install`. Other versions will probably work too. I tested it with Python 3.5.2 using Ubuntu 16.04.

### Pi requirements:
- GNU/Linux 8 (Jessie): http://downloads.raspberrypi.org/raspbian/images/raspbian-2017-07-05
- to run the benchmark script only TensorFlow is required. I installed it using this tutorial: https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi and used the Python 3.4 part:
```
sudo apt-get install libblas-dev liblapack-dev python-dev \
 libatlas-base-dev gfortran python-setuptools
sudo ​pip install \
 http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/lastSuccessfulBuild/artifact/output-artifacts/tensorflow-1.4.0-cp34-none-any.whl
 ```

To create the whole submission file (`make_submission_on_rpi.py`) you'll need these additional packages (`pip3 install` and I got a `HTTPError` which can be solved by using the  `--default-timeout=100` flag):
- pandas==0.22.0
- tqdm==4.19.5
- scipy==1.0.0 (`sudo pip3 install --default-timeout=100 --no-deps scipy`)

Note that installing scipy will take a couple hours. I tried removing this dependency by using the `wave` and `struct` modules but then reading the wavs is super slow (2 hours for all test files).
I tested it with Raspbian GNU/Linux 8 (Jessie) and Python 3.4.2.

## A3.) How To Generate the Solution
## Structure
This repo contains all the code (`.py` or `.ipynb`) to reproduce my part of our submission. You'll find various model definitions in `model.py`, the training script is `train.py` and the scripts to make the submissions are `make_submission.py` (faster as it processes samples in batches) or `make_submission_on_rpi.py` (suitable to create the submission file on the Pi 3: frozen graph, batch size of 1 and fewer dependecies). Keras models checkpoints can be found in `checkpoints_*`, TensorBoard training logs in `logs_*` and frozen TensorFlow graphs in `tf_files`. I am providing these files for the experiments that are required for the final submission. Though, I ran many more. As a result each section of this writeup can be executed independently.
The data is assumed to be in `data/train` and `data/test`:
```
mkdir data
ln -s your/path/to/train data/train
ln -s your/path/to/test data/test
```

## Model training
Only the predicted probabilites by the model from experiment 106 made it to our final ensemble submission.
I trained a lot more but this one was the best. This model is trained with pseudo labels and additional noise. So the first step is to reproduce these samples:
```
git checkout 6892d80
git checkout master REPR_106_pseudo.py submission_091_leftloud_tta_all_labels.csv submission_096_leftloud_tta_all_labels.csv submission_098_leftloud_tta_all_labels.csv
python3 generate_noise.py
python3 REPR_106_pseudo.py
```
And then train the model:
```
mkdir checkpoints_106
python3 train.py
```
For the submission, I selected the checkpoint with the highest validation accuracy using TensorBoard:
```
tensorboard --logdir logs_106
```
The reference model is `checkpoints_106/ep-062-vl-0.1815.hdf5`.

 Note that due to stochasticity (random initialization, data augmentation and me not setting a seed) exactly reproducing these weights is probably not possible.

## Make the submission using TTA:
```
git checkout master make_submission.py
git checkout master checkpoints_106/ep-062-vl-0.1815.hdf5  # change line 64 of `make_submission.py` instead of this command if you use another checkpoint
python3 make_submission.py
```

The resulting submission (`REPR_submission_106_tta_leftloud.csv`) will have a private/public score of 0.88558/0.88349. Every sample is used three times (unchanged, shifted to the left by 1500 timesteps and made louder by multiplying with 1.2). The resulting probabilities are then averaged. Note that this model uses 32 classes. These probabilities will be stored in `REPR_submission_106_tta_leftloud_all_labels_probs.csv`. In order to use them for the ensembled model the order of the samples and the probabilities have to be converted to 12 classes:
```
git checkout master convert_from_see_v3_bugfix.py
python3 convert_from_see_v3_bugfix.py
```
This will create the file: `submission_106_tta_leftloud_all_labels_probs.uint8.memmap` for our ensemble model.

## Raspberry Pi model
This model is trained with pseudo labels from our best ensembled submission: `submission_50.csv`/`submit_50_probs.uint8.memmap`. To train this model, first reproduce the pseudo labels using this submission and then run the training script:
```
git checkout ad2af0d
git checkout master create_pseudo_with_thresh.py submission_50.csv submit_50_probs.uint8.memmap
python3 create_pseudo_with_thresh.py
mkdir checkpoints_195
python3 train.py
```
For this training I am only saving the checkpoints with the best validation accuracy. Therefore, there is no need to inspect the logs. Just use the latest checkpoint.

To freeze the model run:
```
git checkout master freeze_graph.py
git checkout master checkpoints_195/ep-085-vl-0.2231.hdf5  # skip this if you trained the model yourself
python3 freeze_graph.py --checkpoint_path checkpoints_195/ep-085-vl-0.2231.hdf5 --frozen_path tf_files/frozen_195.pb
```
You can then reproduce the best scoring Raspberry Pi submission `rpi_submission_195.csv` by running:
```
# on the pi
git checkout master
python3 make_submission_on_rpi.py --frozen_graph tf_files/frozen_195.pb --test_data data/test/audio --submission_fn rpi_submission_195.csv
```
This submission will have a score of 0.89747/0.90825 on the public/private leaderboard.

## Benchmarking the Pi model
You can benchmark the frozen graph using the provided benchmark binary:

```
curl -O https://storage.googleapis.com/download.tensorflow.org/deps/pi/2017_10_07/benchmark_model
chmod +x benchmark_model
./benchmark_model --graph=tf_files/frozen_195.pb --input_layer="decoded_sample_data:0,decoded_sample_data:1" --input_layer_shape="16000,1:" --input_layer_type="float,int32" --input_layer_values=":16000" --output_layer="labels_softmax:0" --show_run_order=false --show_time=false --show_memory=false --show_summary=true --show_flops=true
```

I got the following results:
- avg time (ms): 58.042
- max memory (bytes): 2180436
- size (bytes): 4870144 (`du -s -B1 tf_files/frozen_195.pb`)
