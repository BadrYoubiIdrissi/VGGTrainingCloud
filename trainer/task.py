import os
import logging
from datetime import datetime
# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable warnings from h5py
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Audio processing and DL frameworks 
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .dataGenerator import Generator

from .model import build_model

import keras
from keras import callbacks
import tensorflow as tf
import argparse
from tensorflow.python.lib.io import file_io

# Constants
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
genresRev = ['metal', 'disco', 'classical', 'hiphop', 'jazz', 
          'country', 'pop', 'blues', 'reggae', 'rock']
num_genres = len(genres)
batch_size = 100
exec_time = datetime.now()

def read_filenames_and_labels(job_dir):
    filenames = file_io.get_matching_files(job_dir+"genres/**/*.*")
    labels = [genres[x.split('/')[-2]] for x in filenames]
    return filenames, labels

def main(job_dir, **args):
    logs_path = job_dir + 'logs/tensorboard'

    filenames, labels = read_filenames_and_labels(job_dir)

    # Transform to a 3-channel image
    fn_train, fn_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.3, random_state=42, stratify = labels)

    train_batch_gen = Generator(fn_train, y_train, batch_size, window=2, fs=16128, nperseg=256)

    test_batch_gen = Generator(fn_test, y_test, batch_size, window=2, fs=16128, nperseg=256)

    # Training step
    input_shape = (129,253)
    cnn = build_model(input_shape, num_genres)
    cnn.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

    #tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    cnn.fit_generator(generator=train_batch_gen, 
                        steps_per_epoch=len(fn_train)//batch_size,
                        epochs=50,
                        verbose=1,
                        workers=8)
    # Evaluate
    score = cnn.evaluate_generator(test_batch_gen, len(fn_test)//batch_size, verbose = 0)
    print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

    # Save the model
    cnn.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model{}.h5'.format(exec_time), mode='w+') as output_f:
                output_f.write(input_f.read())

##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)