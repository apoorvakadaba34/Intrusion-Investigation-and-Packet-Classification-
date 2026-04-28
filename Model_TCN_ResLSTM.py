import numpy as np
import tensorflow as tf
from keras.layers import Input, Activation
from Evaluation import Evaluation
import keras.backend as K
from keras.layers import AveragePooling2D, Permute, Lambda, multiply
from keras.layers import Conv1D, Conv2D, BatchNormalization, ReLU
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam


def coordinateAttentionLayer(x, inputChannel, outputChannel, reductionRatio=32):
    def h_swish(x):
        return ReLU(6., name="ReLU6_1")(x + 3) / 6

    # Hope input x has a shape of NHWC
    identity = x
    [n, h, w, c] = x.shape
    x_h = AveragePooling2D(pool_size=(h, 1), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_w = AveragePooling2D(pool_size=(1, w), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_w = Permute((2, 1, 3))(x_w)
    y = K.concatenate((x_h, x_w), axis=2)
    reductionChannel = max(8, inputChannel // reductionRatio)
    y = Conv2D(filters=reductionChannel, kernel_size=1,
               strides=1, padding="valid")(y)
    y = BatchNormalization()(y)
    y = h_swish(y)
    x_h, x_w = Lambda(tf.split, arguments={"axis": 2, "num_or_size_splits": [w, h]})(y)
    x_w = Permute((2, 1, 3))(x_w)

    a_h = Conv2D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_h)
    a_w = Conv2D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_w)
    a_h = tf.tile(a_h, [1, h, 1, 1])
    a_w = tf.tile(a_w, [1, 1, w, 1])
    out = multiply([identity, a_w, a_h])
    return out


# Define TCN model
def TCN_Feat(Train_Data, num_classes):
    TrainX = np.reshape(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    input_layer = Input(shape=(1, TrainX.shape[2]))

    # Temporal Convolutional Network (TCN) Block
    # Here we use dilation rates that double with each layer
    x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(input_layer)
    x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(x)
    x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=4, activation='relu')(x)
    x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=8, activation='relu')(x)
    # Apply Coordinate Attention
    x = coordinateAttentionLayer(x, inputChannel=64, outputChannel=64)
    # Output layer
    x = Activation('relu')(x)
    x = Conv1D(num_classes.shape[1], kernel_size=1, padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output_layer = Activation('sigmoid')(x)  # Sigmoid for binary classification

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(TrainX, num_classes, epochs=50)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 6
    test = TrainX[:][np.newaxis, ...]
    test = test[:]
    test = np.asarray(test).astype(np.float32)
    layer_out = np.asarray(functors[layerNo]([test])).squeeze()
    return layer_out


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


def Model_TCN_ResLSTM(Data, Target, Epochs=50, Learning_Rate=0.1, batchsize=4, steps_per_epoch=100):
    Feat = TCN_Feat(Data, Target)
    learnperc = round(Feat.shape[0] * 0.75)
    Train_Data = Feat[:learnperc, :]
    Train_Target = Target[:learnperc, :]
    Test_Data = Feat[learnperc:, :]
    Test_Target = Target[learnperc:, :]
    Train_x = np.reshape(Train_Data, (Train_Data.shape[0], Train_Data.shape[1], 1))
    Test_x = np.reshape(Test_Data, (Test_Data.shape[0], Test_Data.shape[1], 1))
    model = build_res_lstm_model((Train_Data.shape[1], 1), Train_Target.shape[1])

    model.compile(optimizer=Adam(learning_rate=Learning_Rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    trainX = np.asarray(Train_x).astype(np.float32)
    testX = np.asarray(Test_x).astype(np.float32)
    model.fit(trainX, Train_Target, epochs=Epochs, batch_size=batchsize, steps_per_epoch=steps_per_epoch)
    pred = model.predict(testX)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = Evaluation(Test_Target, pred)
    return Eval, pred

