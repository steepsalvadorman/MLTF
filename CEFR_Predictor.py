import streamlit as st
from io import StringIO

from cefr_predictor.inference import Model

MAX_FILES = 5
ALLOW_FILES_UPLOADS = False

model = None


def load_model():
    return Model("cefr_predictor/models/logistic_regression.joblib")


def app():
    st.write("# Predictor de dificultad de textos en español")

    textbox_text = st.text_area("Coloca el texto aqui:", height=200)

    if ALLOW_FILES_UPLOADS:
        uploaded_files = st.file_uploader(
            f"Or choose 1 to {MAX_FILES} text files to upload",
            type=["txt"],
            accept_multiple_files=True,
        )
    else:
        uploaded_files = []

    if st.button("Predecir") or textbox_text:

        if len(uploaded_files) > MAX_FILES:
            st.write(f"Too many files. The maximum is {MAX_FILES}.")

        else:
            input_texts = collect_inputs(textbox_text, uploaded_files)

            if input_texts:
                output = model.predict_decode(input_texts)
                display_results(input_texts, output)

            else:
                st.write("Input one or more texts to generate a prediction.")


def collect_inputs(textbox_text, uploaded_files):
    inputs = []

    if textbox_text:
        inputs.append(textbox_text)

    if uploaded_files:
        for upload in uploaded_files:
            stringio = StringIO(upload.getvalue().decode("utf-8"))
            text = stringio.read()
            inputs.append(text)

    return inputs


def display_results(texts, output):
    levels, scores = output

    if ALLOW_FILES_UPLOADS:
        for i, (text, level, score) in enumerate(zip(texts, levels, scores)):
            st.write(f"### Text {i+1}:")
            st.write(f"_{text}_")
            st.write(f"### Nivel de CEFR predecido: __{level}__")
            st.write("### Puntaje por nivel:")
            st.write(score)
    else:
        for level, score in zip(levels, scores):
            st.write(f"### Nivel de CEFR predecido: __{level}__")
            st.write("### Puntaje por nivel:")
            st.write(score)


if __name__ == "__main__":
    model = load_model()
    app()