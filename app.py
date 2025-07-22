from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# === Settings ===
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 5
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Prepare data generators for training ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Build model ===
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(train_gen.class_indices), activation='softmax')  # number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("⏳ Training model, please wait...")

model = create_model()
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("✅ Model training completed.")

# Map class indices back to labels
class_labels = {v: k for k, v in train_gen.class_indices.items()}

# === Flask routes ===
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    img_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_path = filepath

            # Preprocess the image for prediction
            img = load_img(filepath, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_class_index = np.argmax(preds)
            prediction = class_labels[predicted_class_index]

    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
