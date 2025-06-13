import streamlit as st
import os
import librosa
import google.genai as genai
from google.genai import types
from transformers import pipeline
import dotenv

dotenv.load_dotenv()

api_key = os.getenv('APIKEY')


systemInstruction = "You are a translator specialized in interpreting informal spoken Singlish as used by elderly Singaporeans, including words and grammar from Hokkien, Cantonese, Malay, Tamil, and English. Your task is to translate mixed-dialect Singlish sentences into fluent, clear, and natural Standard English while preserving the original speaker's tone, intention, and context. You should resolve vague references (e.g., “that one” or “he like that”) when possible, interpret cultural expressions and interjections like “lah”, “lor”, or “aiyo” into equivalent meanings, and restructure broken grammar into coherent English. Do not translate word-for-word—focus on meaning, tone, and clarity. Assume the speaker is an elderly Singaporean using common informal speech patterns. Your output should be one clean, readable English sentence or paragraph that captures the essence of the original message. Do not say anything else but the translated message. If the original message has no singlish elements, simply repeat the input. If the original message has singlish that you do not know of, please try to intepret how it will sound and translate from there. The user will input the message to be translated"

st.title("Singlish Translator")

@st.cache_resource()
def createPipeline():
    pipe = pipeline("automatic-speech-recognition", model="jensenlwt/whisper-small-singlish-122k")
    client = genai.Client(api_key = api_key)
    return pipe, client

pipe, client = createPipeline()

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
        response = client.models.generate_content(model = 'gemini-2.0-flash',
                                            config=types.GenerateContentConfig(system_instruction=systemInstruction),
                                            contents=transcription)
        st.chat_message('user').write(response.text)
        st.session_state.chat_history.append(('user', response.text))
