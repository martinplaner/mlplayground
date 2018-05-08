from __future__ import print_function

import keras
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize

model = keras.models.load_model('model_cnn.hdf5')
model.summary()


def getData(filename):
    data = imread(filename, flatten=True)
    data = imresize(data, size=(28, 28))
    data = np.invert(data)
    return data.reshape((1, 28, 28, 1))


for f in ['doodles/cat/cat.png', 'doodles/shark/shark.png', 'doodles/bird/bird.png']:
    data = getData(f)
    result = model.predict(data)
    print(f, result)
