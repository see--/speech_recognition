# TensorFlow Speech Recognition Challenge
This repo contains the code for our winning entry in the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) hosted by [kaggle](https://www.kaggle.com).

# Requirements:
- tensorflow-gpu==1.4.0 
- Keras==2.1.2 (the version that comes with tensorflow `2.0.8-tf` should be fine as well but you'll need to adjust the imports)
- tqdm==4.11.2
- scipy==1.0.0
- numpy==1.13.3

All packages can be installed via `pip3 install`. Other versions will probably work too. I tested it with Python 3.5.2 using Ubuntu 16.04.

## Model training
Only the predicted probabilites by the model from experiment 106 made it to our final ensemble submission.
I trained a lot more but this one was the best. This model is trained with pseudo labels. So the first step is to reproduce them:
`git checkout 6892d80`
Then run the Notebook cell that produces the pseudo labels. Later in the competition this step is replaced by the `create_pseudo_with_thresh.py` script. To train the model run:
`python3 train.py`.
For the submission I selected the checkpoint with the highest validation accuracy using tensorboard: `tensorboard --logdir logs_106`. The reference model is `checkpoints_106/ep-062-vl-0.1815.hdf5`.

## Make the submission using TTA
`python3 make_submission.py`: The resulting submission will have a private/public score of 0.88558/0.88349. Every sample is used three times (unchanged, shifted to the left by 1500 timesteps and made louder by multiplying with 1.2). The resulting probabilities are then averaged. Note that this model uses 32 classes. These probabilities will be stored in `REPR_submission_106_tta_leftloud_all_labels_probs.csv`. In order use them for the ensembled model the order of the samples and the probabilities have to be converted: `python3 convert_from_see_v3_bugfix.py`.

## Raspberry Pi model

Overall training the networks was always influenced by stochasticity.
Therefore the trained model weights will vary a bit due to random initialization of the network's weights and random augmentations.


To freeze the model run: `python3 freeze_graph.py --checkpoint_path checkpoints_195/ep-085-vl-0.2231.hdf5`. By default, the frozen graph will be saved as `tf_files/frozen.pb`.

## Benchmarking the Pi model

```curl -O https://storage.googleapis.com/download.tensorflow.org/deps/pi/2017_10_07/benchmark_model chmod +x benchmark_model ./benchmark_model --graph=tf_files/frozen.pb --input_layer="decoded_sample_data:0,decoded_sample_data:1" --input_layer_shape="16000,1:" --input_layer_type="float,int32" --input_layer_values=":16000" --output_layer="labels_softmax:0" --show_run_order=false --show_time=false --show_memory=false --show_summary=true --show_flops=true```