import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.src.optimizers import Adam

from Evaluation import Evaluation


def LSTM_train(trainX, trainY, testX, testY, Epochs, Learning_Rate, steps_per_epoch):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=Learning_Rate), metrics=['accuracy'])
    train = np.asarray(trainX).astype(np.float32)
    test = np.asarray(testX).astype(np.float32)
    model.fit(train, trainY, epochs=Epochs, steps_per_epoch=steps_per_epoch)
    pred = model.predict(test)
    return pred, model


def Model_LSTM(train_data, train_target, test_data, test_target, Epochs=50, Learning_Rate=0.1, steps_per_epoch=100):
    out, model = LSTM_train(train_data, train_target, test_data, test_target, Epochs, Learning_Rate, steps_per_epoch)
    pred = np.asarray(out)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = Evaluation(test_target, pred)

    return Eval, pred

