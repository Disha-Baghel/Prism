import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_model
import numpy as np

X_train = np.load('../data/processed/X_train.npy')
y_train = np.load('../data/processed/y_train.npy')


model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
