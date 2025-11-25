import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D

# from model.metrics import precision, recall, f1_score, top_2_categorical_accuracy
from metrics import precision, recall, f1_score, top_2_categorical_accuracy

def cnn_model(input_shape, num_of_classes):
    model = Sequential()
    model.add(
        BatchNormalization(
            input_shape=input_shape, axis=-1, momentum=0.99, epsilon=0.001
        )
    )
    model.add(
        Conv2D(
            10,
            kernel_size=(10, 10),
            strides=5,
            padding="same",
            input_shape=input_shape,
        )
    )
    convout1 = Activation("relu")
    model.add(convout1)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(20, kernel_size=(10, 10), strides=5, padding="same"))
    convout2 = Activation("relu")
    model.add(convout2)
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            top_2_categorical_accuracy,
            f1_score,
            precision,
            recall,
        ],
    )
    print(model.summary())
    return model
