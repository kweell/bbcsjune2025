import streamlit as st
import os
import librosa
import google.genai as genai
from google.genai import types
from transformers import pipeline

st.title("Singlish Translator")

@st.cache_resource()
def createPipeline():
    pipe = pipeline("automatic-speech-recognition", model="jensenlwt/whisper-small-singlish-122k")
    return pipe

pipe = createPipeline()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

aud = st.audio_input(label='')

for message in st.session_state.chat_history:
    st.chat_message(message[0]).write(message[1])
    print(message)

if aud:
    os.makedirs("audiofiles", exist_ok=True)

    filename = os.path.join("audiofiles", "recorded_audio.wav")

    with open(filename, "wb") as f:
        f.write(aud.getvalue())
    audio, rate = librosa.load(os.path.join("audiofiles", "recorded_audio.wav"), sr=16000)
    result = pipe(audio, generate_kwargs={"language": "english"})
    transcription = result['text']
    if transcription != " **":
        st.chat_message('user').write(transcription)
        st.session_state.chat_history.append(('user', transcription))
    print(transcription)
