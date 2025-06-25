import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ----------- Configuration -------------
#MODEL_PATH = "D:/Semesters Material/FYP/marrowInsight/Code/final_model_inception.keras"
MODEL_PATH = "C:/Users/Ingenious Track/Downloads/multi_class_model_finetuned (1).keras"
IMAGE_DIR = "D:/test_images2"  # <--- replace this with your test folder
IMAGE_SIZE = (160, 160)

# The class order used during training
class_labels = ['ART', 'BLA', 'EBO', 'LYT', 'NGS', 'PMO']

# ----------- Load Model ----------------
model = load_model(MODEL_PATH)

# ----------- Preprocessing Function ----
def preprocess_image(img_path, target_size=IMAGE_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------- Predict All Images --------
results = []
class_frequencies = defaultdict(int)

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(IMAGE_DIR, filename)
        img = preprocess_image(file_path)
        pred = model.predict(img, verbose=0)
        pred_class = class_labels[np.argmax(pred)]
        confidence = np.max(pred)

        # Store result
        results.append({
            "filename": filename,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4)
        })

        # Update class count
        class_frequencies[pred_class] += 1

        print(f"{filename} → {pred_class} ({confidence * 100:.2f}%)")

# ----------- Print Summary -------------
print("\n📊 Class Frequencies:")
for cls in class_labels:
    print(f"{cls}: {class_frequencies[cls]}")

# ----------- Optional: Save to CSV -----
df = pd.DataFrame(results)
df.to_csv("batch_predictions.csv", index=False)
print("\n✅ Predictions saved to 'batch_predictions.csv'")
