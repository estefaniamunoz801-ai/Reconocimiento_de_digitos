import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Reconocimiento de Dígitos", page_icon="🔢")

st.title("🔢 Reconocimiento de Dígitos - MNIST")

# Cargar modelo entrenado en Colab
archivo_modelo = st.file_uploader("Sube tu modelo (.h5)", type=["h5"])

if archivo_modelo is not None:
    with open("modelo_temp.h5", "wb") as f:
        f.write(archivo_modelo.getbuffer())
    modelo = keras.models.load_model("modelo_temp.h5")
    st.success("Modelo cargado correctamente ✅")

    st.write("Dibuja un dígito (0-9):")

    canvas = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predecir"):
        if canvas.image_data is not None:
            # Preprocesamiento: 28x28, escala de grises, normalizar, aplanar
            imagen = Image.fromarray((canvas.image_data[:, :, :3]).astype("uint8"))
            imagen = imagen.convert("L").resize((28, 28))
            arreglo = np.array(imagen).astype("float32") / 255.0
            entrada = arreglo.reshape(1, 784)

            prediccion = modelo.predict(entrada, verbose=0)
            digito = np.argmax(prediccion)
            confianza = np.max(prediccion)

            st.subheader(f"Predicción: {digito}")
            st.write(f"Confianza: {confianza:.2%}")
        else:
            st.warning("Dibuja un dígito antes de predecir.")
else:
    st.info("Sube el archivo modelo_mnist.h5 generado en Colab para comenzar.")
