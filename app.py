import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
from gtts import gTTS
import json
from typing import Optional
from groq import Groq
from pydantic import BaseModel
import os
#from dotenv import load_dotenv # uncomment if you want to use .env file
#load_dotenv()

st.set_page_config(page_title='Groq Translator', 
                   page_icon='🎤',
                   )

st.title('Groq Translator')

# GROQ_API_KEY = "gsk_"
# groq_key = os.getenv("GROQ_API_KEY")
# or 

groq_key = os.getenv("GROQ_API_KEY") if os.getenv("GROQ_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
with st.markdown("🔐 OpenAI"):
    groq_key = st.text_input("Introduce your GROQ_API_KEY (https://console.groq.com/keys/)", value=groq_key, type="password")


client = Groq(api_key=groq_key)

class Translation(BaseModel):
    text: str
    comments: Optional[str] = None


def groq_translate(query, from_language, to_language):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that translates text from {from_language} to {to_language}."
                           f"You will only reply with the translation text and nothing else in JSON."
                           f" The JSON object must use the schema: {json.dumps(Translation.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Translate '{query}' from {from_language} to {to_language}."
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=1024,
        stream=False,
        response_format={"type": "json_object"},
    )
    return Translation.model_validate_json(chat_completion.choices[0].message.content)


model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))

# Speech to text
def speech_to_text(audio_chunk):
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    speech_text = " ".join([segment.text for segment in segments])
    return speech_text

# Text to speech
def text_to_speech(translated_text, language):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text, lang=language)
    my_obj.save(file_name)
    return file_name

languages = {
            "Portuguese": "pt",
            "Spanish": "es",
            "German": "de",
            "French": "fr",
            "Italian": "it",
            "Dutch": "nl",
            "Russian": "ru",
            "Japanese": "ja",
            "Chinese": "zh",
            "Korean": "ko"
            }



def main():
    st.write('Speak in your native language and select the translation you need')
    option = st.selectbox(
                        "Language to translate to:",
                        #("Portuguese", "Spanish", "German", "French", "Italian", "Dutch", "Russian", "Japanese", "Chinese", "Korean"),
                        #{"Portuguese": "pt", "Spanish": "es", "German": "de", "French": "fr", "Italian": "it", "Dutch": "nl", "Russian": "ru", "Japanese": "ja", "Chinese": "zh", "Korean": "ko"},
                        languages,
                        index=None,
                        placeholder="Select language...",
                        )
    # Record audio
    audio_bytes = audio_recorder()
    if audio_bytes and option:
        # Display audio player
        st.audio(audio_bytes, format="audio/wav")
        with open('audio.wav', mode='wb') as f:
            f.write(audio_bytes)

        # Speech to text
        st.divider()
        with st.spinner('Transcribing...'):
            text = speech_to_text('audio.wav')
        st.subheader('Transcribed Text')
        st.write(text)

        # Groq translation
        st.divider()
        with st.spinner('Translating...'):
            translation = groq_translate(text, 'en', option)
        st.subheader('Translated Text to ' + option)
        st.write(translation.text)

        # Text to speech di gTTS
        audio_file = text_to_speech(translation.text, languages[option])
        st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()