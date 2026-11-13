from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re

"""
1. make some constants
a. model path (Bigru.keras)
b. tokenizer path 
c. max_seq_length 
d. emotion labels 
e. emotion denotion by emojis  
"""
model_path = "Artifacts/Bigru_model.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"
max_sequence_length = 50
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

"""
2. preprocess the upcoming text
a. cleans raw text so it matches the format used while training 
a. convert the text to lowercase
b. remove apostrophes
c. remove special characters
d. remove extra spaces 

"""
def preprocess_text(text : str) -> str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text

"""
3. request and response schema 
a. input text
b. prediction response 
c. health response (server health)
"""

class TextInput(BaseModel):
    text : str = Field(..., min_length=1, max_length=2000, description="the sentence to analyze", json_schema_extra={"example":"i feel so happy!"})

class PredictionResponse(BaseModel):
    text : str
    predicted_emotion : str
    confidence : float
    all_probabilities : dict[str , float]

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status : str
    model_laoded : bool 
    
"""
4. model loading and lifespan management 
a. load the model and tokenizer once the server starts up 
b. 
"""
dl_model = {}
@asynccontextmanager
async def lifespan(app : FastAPI):
    print("loading the model and tokenizer..")
    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)

    print("Model are loaded successfully!!")

    yield
    print("unloading the model")
    dl_model.clear()

app = FastAPI(
    lifespan=lifespan
)

"""
5. Mount the static files to the fastapi app 
a. Enable cor origin to allow requests from different origins.

"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static',StaticFiles(directory="static"), name = "static")


"""
6. API endpoints.
a. server UI at homepage('/')
b. health cehck-up endpoint('/health')
c. predict emotion endpoint('/predict')
"""

@app.get('/', include_in_schema=False)
def serve_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="server is running", model_laoded=bool(dl_model))

@app.post('/predict',response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    # cleans the input sentence 
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code = 503, detail = "model is not loaded yet please try again later")

    cleaned_text = preprocess_text(text_input.text)

    # convert the words into numeric using tokenizer
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])

    # pad the sequence to ensure uniform length 
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )
    # run prediction using BiGRU model 
    probabilites = BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probabilites)) # 4
    all_probabilites =  {
        label: float(prob) for label, prob in zip(emotion_labels, probabilites)
    }
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilites[top_emotion_index]),
        all_probabilities= all_probabilites
    )
    # return the top emotion and full probability breakdown  

