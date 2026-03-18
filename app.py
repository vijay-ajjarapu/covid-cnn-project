import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("COVID X-ray Prediction")

model = tf.keras.models.load_model("model.h5")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("COVID Positive")
    else:
        st.success("COVID Negative")