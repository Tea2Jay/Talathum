import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
from time import sleep
from tensorflow.python.keras.backend import dtype

from tensorflow.python.keras.utils.generic_utils import to_list
from realTimeLatentWalk import LatentWalkerController

# controller = LatentWalkerController()


# Hide GPU from visible devices
tf.config.set_visible_devices([], "GPU")


def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(
        range(1, len(model_history.history["accuracy"]) + 1),
        model_history.history["accuracy"],
    )
    axs[0].plot(
        range(1, len(model_history.history["val_accuracy"]) + 1),
        model_history.history["val_accuracy"],
    )
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(
        np.arange(1, len(model_history.history["accuracy"]) + 1),
        len(model_history.history["accuracy"]) / 10,
    )
    axs[0].legend(["train", "val"], loc="best")
    # summarize history for loss
    axs[1].plot(
        range(1, len(model_history.history["loss"]) + 1), model_history.history["loss"]
    )
    axs[1].plot(
        range(1, len(model_history.history["val_loss"]) + 1),
        model_history.history["val_loss"],
    )
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_xticks(
        np.arange(1, len(model_history.history["loss"]) + 1),
        len(model_history.history["loss"]) / 10,
    )
    axs[1].legend(["train", "val"], loc="best")
    fig.savefig("plot.png")
    plt.show()


# Define data generators
# train_dir = 'data/train'
# val_dir = 'data/test'

# num_train = 28709
# num_val = 7178
# batch_size = 64
# num_epoch = 50

# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

# emotions will be displayed on your face from the webcam feed
model.load_weights("model.h5")
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
# emotion_dict = {0: "Angry", 1: "Happy", 2: "Sad"}
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}


# start the webcam feed
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture("C:/Users/3izzo/Desktop/Projects/Talathum/WIN_20211021_23_20_21_Pro.mp4")
facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# sleep(20)
# sleep(1)

emotions = []
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
        )
        prediction = model.predict(cropped_img)
        prediction = prediction[0].tolist()
        emotions.append(prediction)
        if len(emotions) >= 5:
            avg = [0, 0, 0, 0, 0, 0, 0]
            avg = np.sum(emotions,0).tolist()
            avg = np.divide(avg, 5).tolist()
            maxindex = int(np.argmax(avg))
            cv2.putText(
                frame,
                emotion_dict[maxindex],
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            print(f'{emotions=}')
            emotions.pop(0)
            print(f'after pop {emotions=}')

        # prediction = prediction[0]
        # prediction = [prediction[0], prediction[3], prediction[5]]
        # emotions = [emotions[0]+prediction[0], emotions[1] +
        #             prediction[1], emotions[2]+prediction[2]]

    cv2.imshow("Video", cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
exit()
