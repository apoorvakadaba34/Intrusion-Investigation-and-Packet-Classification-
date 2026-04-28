import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import tensorflow as tf
from Evaluation import Evaluation


# Residual LSTM block
def resblock(inputs, filters, strides):
    y = inputs  # Shortcut path

    # Main path
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Fit shortcut path dimenstions
    if strides > 1:
        y = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding='same',
        )(y)
        y = tf.keras.layers.BatchNormalization()(y)

    # Concatenate paths
    x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.ReLU()(x)

    return x


def build_res_lstm_model(input_shape, num_classes, units=64, num_blocks=3):
    inputs = Input(shape=input_shape)

    x = inputs
    for _ in range(num_blocks):
        x = resblock(x, 64, 1)

    x = LSTM(units)(x)  # Final LSTM layer without return_sequences
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def Model_Res_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Epochs=50, Learning_Rate=0.1, steps_per_epoch=100):
    Train_x = np.reshape(Train_Data, (Train_Data.shape[0], Train_Data.shape[1], 1))
    Test_x = np.reshape(Test_Data, (Test_Data.shape[0], Test_Data.shape[1], 1))
    model = build_res_lstm_model((Train_Data.shape[1], 1), Train_Target.shape[1])

    model.compile(optimizer=Adam(learning_rate=Learning_Rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    trainX = np.asarray(Train_x).astype(np.float32)
    testX = np.asarray(Test_x).astype(np.float32)
    model.fit(trainX, Train_Target, epochs=Epochs, steps_per_epoch=steps_per_epoch)
    pred = model.predict(testX)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = Evaluation(Test_Target, pred)
    return Eval, pred
