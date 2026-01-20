import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# --------------------
# HyperParameters
# --------------------
IMG_HEIGHT, IMG_WIDTH = 96, 96   # reduced image size
SEQUENCE_LENGTH = 30           # frames per video
BATCH_SIZE = 8                   # small to save memory
EPOCHS = 50
NUM_CLASSES = 2

# --------------------
# Custom Video Generator
# --------------------
class VideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size=8, frames=10, size=(96, 96)):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.frames = frames
        self.size = size

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = []
        for path in batch_x:
            frames = self._load_video(path)
            X.append(frames)

        return np.array(X, dtype=np.float32), np.array(batch_y)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(self.frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.size)
            frame = frame.astype("float32") / 255.0  # normalize
            frames.append(frame)
        cap.release()

        # Pad if fewer frames
        while len(frames) < self.frames:
            frames.append(np.zeros((*self.size, 3), dtype=np.float32))

        return np.array(frames)

# --------------------
# Load dataset paths
# --------------------
violent_dir = ""
nonviolent_dir = ""

violent_videos = [os.path.join(violent_dir, f) for f in os.listdir(violent_dir) if f.endswith(".mp4")]
nonviolent_videos = [os.path.join(nonviolent_dir, f) for f in os.listdir(nonviolent_dir) if f.endswith(".mp4")]

video_paths = violent_videos + nonviolent_videos
labels = [1] * len(violent_videos) + [0] * len(nonviolent_videos)

print("Total videos:", len(video_paths))

# Train/test split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    video_paths, labels, test_size=0.2, random_state=42
)

train_gen = VideoGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, frames=SEQUENCE_LENGTH, size=(IMG_HEIGHT, IMG_WIDTH))
val_gen   = VideoGenerator(test_paths, test_labels, batch_size=BATCH_SIZE, frames=SEQUENCE_LENGTH, size=(IMG_HEIGHT, IMG_WIDTH))

# --------------------
# Build Model (CNN + LSTM)
# --------------------
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
mobilenet.trainable = True

for layer in mobilenet.layers[:-44]:  # freeze earlier layers
    layer.trainable = False

x = mobilenet.output
x = GlobalAveragePooling2D()(x)
cnn_model = Model(inputs=mobilenet.input, outputs=x)

model = Sequential()
model.add(TimeDistributed(cnn_model, input_shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(LSTM(32, return_sequences=False))  # keep LSTM
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# --------------------
# Compile Model
# --------------------
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, verbose=1, min_lr=0.0005)

# --------------------
# Train Model
# --------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# --------------------
# Plot Accuracy/Loss
# --------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("accuracy_curve.jpg")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("loss_curve.jpg")
plt.show()

# --------------------
# Evaluate
# --------------------
test_loss, test_accuracy = model.evaluate(val_gen)
print(f'Test Accuracy: {test_accuracy:.2f}')

pred = model.predict(val_gen)
pred_binary = (pred > 0.5).astype(int)

cm = confusion_matrix(test_labels, pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.savefig("confusion_matrix.jpg")
plt.show()

print(classification_report(test_labels, pred_binary))
print("Precision:", precision_score(test_labels, pred_binary))
print("Recall:", recall_score(test_labels, pred_binary))
print("F1-Score:", f1_score(test_labels, pred_binary))

model.save('violence_detection_MobileNet_Lstm_model.h5')


