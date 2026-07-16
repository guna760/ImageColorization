from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_model():

    inputs = Input(shape=(128,128,1))

    # Encoder
    x = Conv2D(64,3,padding='same',activation='relu')(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(128,3,padding='same',activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256,3,padding='same',activation='relu')(x)

    x = Conv2D(512,3,padding='same',activation='relu')(x)

    # Decoder
    x = UpSampling2D()(x)
    x = Conv2D(256,3,padding='same',activation='relu')(x)

    x = UpSampling2D()(x)
    x = Conv2D(128,3,padding='same',activation='relu')(x)

    x = Conv2D(64,3,padding='same',activation='relu')(x)

    outputs = Conv2D(2,1,activation='tanh')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    return model