import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pydub import AudioSegment
from audiorecorder import audiorecorder
import io
import numpy as np
import wave
import requests

# Set up Streamlit page configuration
st.set_page_config(layout="wide")

# Set device to mps if available (for Apple Silicon devices)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Mistral API details
MISTRAL_API_KEY = "lHFGShbf91kbUx1vrJ2rqLDIaJnAhYBy"  # Set your API key in environment variables
MISTRAL_API_URL = "https://codestral.mistral.ai/v1/fim/completions"  # Replace with the actual Mistral endpoint

# Check if MISTRAL_API_KEY is set
if not MISTRAL_API_KEY:
    st.warning("Please set your MISTRAL_API_KEY as an environment variable!")

# Initialize Whisper and TTS models
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    
    # Initialize the TTS model
    tts_model = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    
    return processor, model, tts_model

processor, model, tts_model = load_models()

# Transcribe audio to text
def transcribe_audio(audio_bytes):
    try:
        # Decode audio bytes using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)
        raw_audio = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize PCM values

        # Prepare inputs for Whisper model
        inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(inputs)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "Unable to transcribe audio."

# Generate response with Mistral API
def generate_response(transcription):
    if not MISTRAL_API_KEY:
        return "MISTRAL_API_KEY is not set."

    try:
        # Prepare the request payload
        payload = {
            "model": "codestral-latest",  # Replace with the correct model ID
            "prompt": f"You are a helpful assistant. Summarize your response in 80-100 words.\nUser: {transcription}\nAI:",
            "max_tokens": 150,
            "temperature": 0.7,
        }

        # Set up headers for the API request
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

        # Send the request to the Mistral API
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
        
        # Debugging: Log raw response data
        # st.write(response.content)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                return "The API response format is invalid or missing 'choices'."
        else:
            return f"Error {response.status_code}: {response.content.decode('utf-8', errors='ignore')}"
    except Exception as e:
        st.error(f"Error communicating with Mistral API: {e}")
        return "Unable to generate a response."

# Convert text to speech using TTS model and prepare for Streamlit playback
def text_to_speech(response):
    try:
        # Synthesize speech from text
        speech = tts_model(response)
        audio_array = speech["audio"]
        sample_rate = speech["sampling_rate"]

        # Ensure the audio is in the correct range for 16-bit audio
        audio_array = np.clip(audio_array, -1.0, 1.0)  # Clip to -1 to 1 range
        audio_array = (audio_array * 32767).astype(np.int16)  # Scale to 16-bit PCM

        # Convert numpy array to WAV format bytes for Streamlit
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())  # Write as 16-bit PCM

        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Streamlit Interface
st.title("Voice Bot (English)")

# Use audiorecorder to capture audio
audio = audiorecorder(start_prompt="", stop_prompt="")

if len(audio) > 0:
    # Play the recorded audio in frontend
    st.audio(audio.export().read(), format="audio/wav")

    # Automatically transcribe after recording stops
    st.write("Transcribing the recorded audio...")
    audio_bytes = audio.export().read()  # Get audio bytes
    transcription = transcribe_audio(audio_bytes)

    # Display transcription
    st.markdown("### User Prompt:")
    st.markdown(f"##### {transcription}")

    # Generate response from Mistral API
    st.write("Generating the response...")
    response = generate_response(transcription)

    # Display AI assistant's response
    st.markdown("### AI Assistant:")
    st.markdown(f"##### {response}")

    # Convert response to speech and play it
    st.write("Generating speech from response...")
    audio_response = text_to_speech(response)
    # st.write(f"MISTRAL_API_KEY is loaded: {bool(MISTRAL_API_KEY)}")


    if audio_response:
        # Play the TTS audio response
        st.audio(audio_response, format="audio/wav")