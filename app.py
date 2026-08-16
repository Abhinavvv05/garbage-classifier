import os
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# ---- Load the TFLite model ----
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = input_details[0]["shape"][1]  # e.g. 224

# ---- Load labels (handles "0 Glass" style lines from Teachable Machine) ----
with open("labels.txt", "r") as f:
    labels = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)


def classify_image(img):
    if img is None:
        return {}

    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0  # Teachable Machine's standard normalization
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Defensive softmax normalization in case raw output isn't already 0-1
    output = output.astype(np.float64)
    total = output.sum()
    if total > 0 and abs(total - 1.0) > 0.02:
        output = output / total

    return {labels[i]: float(output[i]) for i in range(len(labels))}


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload a garbage image"),
    outputs=gr.Label(num_top_classes=len(labels), label="Predicted Waste Category"),
    title="Garbage Classification",
    description="Upload an image and the model will sort it into one of the trained waste categories.",
)

if __name__ == "__main__":
    demo.launch()
