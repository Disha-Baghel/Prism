import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate

def build_model():
    input_img = Input(shape=(256, 256, 1))

    # Encoder
    conv = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(conv)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same')(conv)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(conv)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same')(conv)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(conv)

    # Decoder
    conv = UpSampling2D((2, 2))(conv)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same')(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same')(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(conv)

    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv)

    return Model(input = input_img, outputs = output_img)