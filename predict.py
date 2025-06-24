import os
import torch
from multitalk.models.multitalk import MultiTalkModel
from multitalk.utils.audio import load_audio
from multitalk.utils.text import postprocess_text

model = None

def load_model():
    global model
    if model is None:
        model = MultiTalkModel.from_pretrained("checkpoints/multitalk.pt")  # adjust this path if needed
        model.eval()

def predict(audio_path: str, task: str = "transcribe"):
    load_model()
    audio = load_audio(audio_path)
    output = model.transcribe(audio, task=task)
    return {"text": postprocess_text(output)}
