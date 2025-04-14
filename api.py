from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import subprocess
import uuid
import os

app = FastAPI()

# Cho phép gọi từ React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Đổi thành domain cụ thể nếu cần
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Biến toàn cục để lưu transcript video (dùng tạm cho demo)
video_transcripts = {}

# ============================ API Models ============================

class ChatRequest(BaseModel):
    question: str

class VideoRequest(BaseModel):
    youtube_link: str

# ============================ Endpoints ============================

@app.post("/video")
async def process_video(req: VideoRequest):
    video_id = str(uuid.uuid4())
    audio_path = f"/tmp/{video_id}.mp3"
    text_path = f"/tmp/{video_id}.txt"

    try:
        # Bước 1: Tải video từ YouTube và tách âm thanh
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, req.youtube_link
        ], check=True)

        # Bước 2: Dùng Whisper để chuyển giọng nói thành văn bản
        subprocess.run([
            "whisper", audio_path, "--model", "base", "--output_format", "txt", "--output_dir", "/tmp"
        ], check=True)

        # Bước 3: Đọc transcript
        with open(text_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        video_transcripts["latest"] = transcript
        return {"message": "Video đã được xử lý", "transcript": transcript}

    except Exception as e:
        return {"message": f"Lỗi xử lý video: {str(e)}"}

@app.post("/chat")
async def chat(req: ChatRequest):
    transcript = video_transcripts.get("latest", "")
    prompt_text = f"{transcript}\n\nCâu hỏi: {req.question}"

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {"answer": response[0]}
