from __future__ import print_function

import h5py
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import Dense
from keras.models import Sequential

batch_size = 256
epochs = 10  # 20

cls = 0
data = []
labels = []
classes = ["apple", "banana", "bird", "cat", "hat", "shark", "star", "table", "truck"]
for f in classes:
    d = np.load('data/' + f + '.npy')
    d = d.reshape((-1, 28, 28, 1))
    l = keras.utils.to_categorical(np.full((len(d),), cls), len(classes))
    data.append(d)
    labels.append(l)
    cls = cls + 1

data = np.concatenate(data)
labels = np.concatenate(labels)

# data = data.astype('float32')
# data /= 255

# print(labels)
print('data shape: ', data.shape)
print('label shape: ', labels.shape)
# print(data.shape[0], 'samples')
# print(data[0])

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.15),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    Flatten(),
    Dropout(0.1),
    Dense(len(classes), activation='softmax'),
])

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(data, labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          shuffle=True,
          verbose=1)

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

outfile = "model_cnn.hdf5"
model.save(outfile)

keras.utils.plot_model(model, to_file='model_cnn.png')
