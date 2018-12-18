import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
#from keras.applications.vgg16 import VGG16

def build_model(input_shape, num_genres, freezed_layers = 5):
    # input_tensor = Input(shape=input_shape)
    # vgg16 = VGG16(include_top=False, weights=None,
    #               input_tensor=input_tensor)

    # top = Sequential()
    # top.add(Flatten(input_shape=vgg16.output_shape[1:]))
    # top.add(Dense(256, activation='relu'))
    # top.add(Dropout(0.5))
    # top.add(Dense(num_genres, activation='softmax'))

    # model = Model(inputs=vgg16.input, outputs=top(vgg16.output))
    # # for layer in model.layers[:freezed_layers]:
    # #     layer.trainable = False

    model = Sequential()
    # Conv Block 1
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                    activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Conv Block 2
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Conv Block 3
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Conv Block 4
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Conv Block 5
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    model.add(Dropout(0.25))

    # MLP
    model.add(Flatten())
    model.add(Dense(num_genres, activation='softmax'))

    return model