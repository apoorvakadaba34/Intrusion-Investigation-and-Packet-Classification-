import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam

from Evaluation import Evaluation


def Model(X, Y, test_x, test_y, Epochs, Learning_Rate, steps_per_epoch):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, X.shape[2])),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(Y.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=Learning_Rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()
    train = np.asarray(X).astype(np.float32)
    test = np.asarray(test_x).astype(np.float32)
    # Train the model
    model.fit(train, Y, epochs=Epochs, steps_per_epoch=steps_per_epoch)
    pred = model.predict(test)

    return pred


def Model_CNN(train_data, train_target, test_data, test_target, Epochs=50, Learning_Rate=0.1, steps_per_epoch=100):
    trainX = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
    testX = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
    pred = Model(trainX, train_target, testX, test_target, Epochs, Learning_Rate, steps_per_epoch)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = Evaluation(pred, test_target)

    return Eval, pred

