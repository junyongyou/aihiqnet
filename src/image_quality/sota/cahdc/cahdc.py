from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, Dropout, Reshape, GlobalAveragePooling2D
from tensorflow.keras.backend import squeeze


def create_model(input_shape, dropout_keep_prob=0.5):
    inputs = Input(shape=input_shape)

    net = Conv2D(32, (3, 3), padding='same')(inputs)
    net = Conv2D(32, (3, 3), padding='same')(net)
    net = MaxPool2D((2, 2), padding='same')(net)

    net = Conv2D(32, (3, 3), padding='same')(net)
    net = Conv2D(32, (3, 3), padding='same')(net)
    net = MaxPool2D((2, 2), padding='same')(net)

    net = Conv2D(64, (3, 3), padding='same')(net)
    net = Conv2D(64, (3, 3), padding='same')(net)
    net = Conv2D(64, (3, 3), padding='same')(net)
    net_3 = Conv2D(64, (1, 1), padding='same')(net)
    net_3 = Conv2D(64, (3, 3), strides=2, padding='same')(net_3)
    net_3 = Conv2D(64, (3, 3), strides=2, padding='same')(net_3)
    net_3 = Conv2D(64, (3, 3), strides=2, padding='same')(net_3)
    net = MaxPool2D((2, 2), padding='same')(net)

    net = Conv2D(64, (3, 3), padding='same')(net)
    net_4 = Conv2D(64, (1, 1), padding='same')(net)
    net_4 = Conv2D(64, (3, 3), strides=2, padding='same')(net_4)
    net_4 = Conv2D(64, (3, 3), strides=2, padding='same')(net_4)
    net = MaxPool2D((2, 2), padding='same')(net)

    net = Conv2D(64, (3, 3), padding='same')(net)
    net_5 = Conv2D(64, (1, 1), padding='same')(net)
    net_5 = Conv2D(64, (3, 3), strides=2, padding='same')(net_5)

    net = Conv2D(512, (3, 3), padding='same')(net)
    net_6 = Conv2D(64, (1, 1), strides=2, padding='same')(net)

    f1 = MaxPool2D((10, 10), (10, 10), padding='valid')(net_3)
    f2 = MaxPool2D((10, 10), (10, 10), padding='valid')(net_4)
    f3 = MaxPool2D((10, 10), (10, 10), padding='valid')(net_5)
    f4 = MaxPool2D((10, 10), (10, 10), padding='valid')(net_6)

    fm = Concatenate(axis=3)([f1, f2, f3, f4])

    qm = Conv2D(100, (1, 1), padding='same')(fm)
    qm = Dropout(dropout_keep_prob)(qm)
    qm = Conv2D(1, (1, 1), padding='same')(qm)
    qm = GlobalAveragePooling2D()(qm)

    model = Model(inputs=inputs, outputs=qm)
    model.summary()
    return model


if __name__ == '__main__':
    create_model((None, None, 3))
