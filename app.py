import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tflite_runtime.interpreter import Interpreter

st.set_page_config(page_title="Garbage Classification", page_icon="🗑️")

@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_labels():
    with open("labels.txt", "r") as f:
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels

interpreter = load_model()
class_names = load_labels()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]["shape"][1]


def classify_image(image: Image.Image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    arr = np.asarray(image).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float64)

    total = output.sum()
    if total > 0 and abs(total - 1.0) > 0.02:
        output = output / total

    return output


st.title("🗑️ Garbage Classification")
st.write("Upload an image and the model will sort it into one of the trained waste categories.")

uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    scores = classify_image(image)
    results = sorted(zip(class_names, scores), key=lambda x: -x[1])

    top_label, top_score = results[0]
    st.subheader(f"Prediction: {top_label} ({top_score*100:.1f}%)")

    st.write("Confidence breakdown:")
    for label, score in results:
        st.write(f"{label}: {score*100:.1f}%")
        st.progress(min(float(score), 1.0))
