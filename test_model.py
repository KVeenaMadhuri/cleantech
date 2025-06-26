from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Load the model
model = load_model('waste_classifier.h5')

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test images
test_dir = 'test'  # Make sure this path exists with correct subfolders
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Print class indices to verify order
print("Test folder class indices:", test_generator.class_indices)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")
print(f"📉 Test Loss: {loss:.4f}\n")

# Predict
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Create reverse label map from original training order
labels = {0: 'paper', 1: 'plastic'}

# Print predictions
filenames = test_generator.filenames
for i in range(len(filenames)):
    print(f"{filenames[i]} ➜ Predicted: {labels[predicted_classes[i]]}")


