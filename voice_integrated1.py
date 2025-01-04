import os
import io
import faiss
import numpy as np
import pandas as pd
from langchain.document_loaders import UnstructuredPDFLoader
from sentence_transformers import SentenceTransformer
import fitz
import streamlit as st
from io import BytesIO
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
from audiorecorder import audiorecorder
import requests
from gtts import gTTS

# Helper Functions
def load_pdf(uploaded_file):
    """Extracts text and metadata from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    metadata = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
        metadata.append({"page_number": page_num + 1, "page_text": page.get_text()})
    return text, metadata

def load_excel(file_path):
    """Extracts text from an Excel file."""
    df = pd.read_excel(file_path)
    return "\n".join([" ".join(map(str, row)) for row in df.values])

def split_into_chunks(text, max_length=512):
    """Splits large text into smaller chunks for embedding."""
    words = text.split()
    chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return list(set(chunks))  # Remove duplicate chunks

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """Generates embeddings for a list of texts."""
    embeddings = model.encode(texts)
    return np.array(embeddings)

# Initialize FAISS Index
embedding_dim = 384  # Match the model's dimensionality
index = faiss.IndexFlatL2(embedding_dim)  # L2 similarity for FAISS
metadata_store = []
text_store = []

def store_in_faiss(embeddings, metadata, texts):
    """Stores embeddings, metadata, and actual texts in FAISS."""
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Embedding dimensionality mismatch: {embeddings.shape[1]} vs {embedding_dim}")
    index.add(embeddings)
    metadata_store.extend(metadata)
    text_store.extend(texts)

def retrieve(query, top_k=5):
    """Retrieves top-k similar documents for a given query."""
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    
    unique_results = set()
    results = []
    
    for idx, i in enumerate(indices[0]):
        if text_store[i] not in unique_results:
            unique_results.add(text_store[i])
            results.append((text_store[i], metadata_store[i], distances[0][idx]))
    
    return results

# Speech Recognition Setup
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

def transcribe_audio(audio_bytes):
    """Transcribes audio bytes to text using Whisper."""
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    audio_data = np.array(audio_segment.get_array_of_samples())
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    generated_ids = model_whisper.generate(inputs["input_features"], forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"))
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Convert text to speech using gTTS
def text_to_speech(response):
    """Converts AI response text to speech using gTTS and prepares for playback in Streamlit."""
    try:
        # Use gTTS to convert text to speech
        tts = gTTS(text=response, lang='en')

        # Save the speech to a temporary file
        audio_file_path = "/tmp/response_audio.mp3"
        tts.save(audio_file_path)

        # Ensure the file exists before using it in Streamlit
        if os.path.exists(audio_file_path):
            return audio_file_path  # Path to the saved audio file

    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Generate response with Mistral API
def generate_response(transcription):
    MISTRAL_API_KEY = ""  # Set your API key in environment variables
    MISTRAL_API_URL = "https://codestral.mistral.ai/v1/fim/completions"  # Replace with the actual Mistral endpoint
    
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

# Streamlit UI
st.title("Document Search and Retrieval with AI Assistant")
st.write("Upload a PDF or Excel file and enter a query to retrieve similar documents.")

# File Upload
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "xlsx"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.write("Processing PDF...")
        document_text, metadata = load_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        st.write("Processing Excel...")
        document_text = load_excel(uploaded_file)
        metadata = [{} for _ in range(len(document_text))]

    # Split document into chunks
    chunks = split_into_chunks(document_text)
    st.write(f"Document split into {len(chunks)} chunks.")

    # Embed chunks
    st.write("Generating embeddings...")
    embeddings = embed_texts(chunks)

    # Store in FAISS
    store_in_faiss(embeddings, metadata, chunks)
    st.write("Embeddings and metadata stored in FAISS.")

    # Voice Input
    st.write("Speak your query (click the microphone icon):")
    audio_bytes = audiorecorder("Click to record", "Recording...")

    if audio_bytes:
        # Convert AudioSegment to bytes
        audio_binary = BytesIO()
        audio_bytes.export(audio_binary, format="wav")
        audio_binary.seek(0)
        st.audio(audio_binary, format="audio/wav")
        
        try:
            query = transcribe_audio(audio_binary.getvalue())

            # Retrieve relevant documents
            results = retrieve(query)
            relevant_text = " ".join([result[0] for result in results])

            # Generate response from Mistral API
            response_text = generate_response(relevant_text)
            st.markdown("### AI Assistant:")
            st.markdown(f"##### {response_text}")

            # Convert AI response to speech
            audio_file_path = text_to_speech(response_text)

            if audio_file_path:
                st.audio(audio_file_path, format='audio/mp3')  # Play the generated audio file in Streamlit

        except Exception as e:
            st.error(f"Error during audio transcription: {e}")
