from keras.utils import to_categorical
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy import signal
import numpy as np
import soundfile
import librosa
import os
from tensorflow.python.lib.io import file_io


class Generator(Sequence):

    def __init__(self, filenames, labels, batch_size, window = 2, fs=16128, nperseg=256):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size
        self.window = window #in seconds
        self.fs = fs
        self.nperseg = nperseg

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = []
        Y = []

        for i in range(self.batch_size):
            x, y = self.load_from_filename(batch_x[i], batch_y[i]) 
            X.append(x)
            Y.append(y)

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        return X,Y 
    
    def load_from_filename(self, filename, label):
        print("Loading file : "+filename)
        tmp = file_io.FileIO(filename, "rb")
        soundSignal, orig_fs = soundfile.read(tmp)
        print("Resampling")
        soundSignal = librosa.resample(soundSignal, orig_fs, self.fs, res_type="kaiser_fast")
        #self.fs = orig_fs
        signals, y = self.splitsongs(soundSignal, label)
        print("Performin stft")
        specs = self.to_stft(signals)
        return specs, to_categorical(np.array(y), num_classes=10, dtype='int32')

    """
    @description: Method to split a song into multiple songs using overlapping windows
    """
    def splitsongs(self, X, y):
        # Empty lists to hold our results
        temp_X = []
        temp_y = []

        # Get the input song array size
        xshape = X.shape[0]
        chunk = int(self.fs*self.window)
        #offset = int(chunk*(1.-self.overlap))
        
        # Split the song and create new ones on windows
        #spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
        truncate = xshape - (xshape % chunk)
        X = X[:truncate]

        spsong = [X[i:i+chunk] for i in range(0,X.shape[0], chunk)]
        for s in spsong:
            temp_X.append(s)
            temp_y.append(y)

        return np.array(temp_X), np.array(temp_y)

    def stft(self, x):
        _, _, Zxx = signal.stft(x, fs=self.fs, nperseg=self.nperseg)
        #return np.stack([20*np.log(abs(Zxx)+1e-5),np.angle(Zxx)], axis=-1)
        return (20*np.log(abs(Zxx)+1e-5))

    def istft(self, x):
        pass
    """
    @description: Method to convert a list of songs to a np array of stft amplitude and phase
    """
    def to_stft(self, songs):
        # Transformation function
        stftMap = lambda x: self.stft(x)

        # map transformation of input songs to stft
        tsongs = list(map(stftMap, songs))
        return np.array(tsongs)