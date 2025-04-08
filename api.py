from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

app = FastAPI()

# Cho phép frontend React gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nếu biết rõ origin có thể thay bằng http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Gửi câu hỏi
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "user", "content": [{"type": "text", "text": req.question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {"answer": response[0]}


# Gửi link video YouTube
class VideoRequest(BaseModel):
    youtube_link: str

@app.post("/video")
async def process_video(req: VideoRequest):
    # TODO: Thêm xử lý nếu bạn đã có logic tách text từ video (gợi ý: sử dụng Whisper + yt-dlp)
    return {"message": f"Video '{req.youtube_link}' đã được xử lý (giả lập)."}
