import numpy as np
import tensorflow as tf
from keras.layers import Input, Activation
from keras.models import Model
from keras.optimizers import Adam
from Evaluation import Evaluation
import keras.backend as K
from keras.layers import AveragePooling2D, Permute, Lambda, multiply
from keras.layers import Conv1D, Conv2D, BatchNormalization, ReLU


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
def build_tcn_model(Train_Data, num_classes):
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
    return model


def Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target, Epochs=50, Learning_Rate=0.1, steps_per_epoch=100):
    model = build_tcn_model(Train_Data, Train_Target)
    model.compile(optimizer=Adam(learning_rate=Learning_Rate), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    TrainX = np.reshape(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    TestX = np.reshape(Test_Data, (Test_Data.shape[0], 1, Test_Data.shape[1]))
    trainX = np.asarray(TrainX).astype(np.float32)
    testX = np.asarray(TestX).astype(np.float32)
    model.fit(trainX, Train_Target, epochs=Epochs, steps_per_epoch=steps_per_epoch)

    pred = model.predict(testX)
    Eval = Evaluation(pred, Test_Target)
    return Eval, pred
