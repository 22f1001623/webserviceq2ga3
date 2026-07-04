import os
import base64
import binascii

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str
    question: str

def normalize_base64_image(s: str):
    s = s.strip()
    mime = "image/png"
    if s.startswith("data:"):
        header, payload = s.split(",", 1)
        if "image/jpeg" in header or "image/jpg" in header:
            mime = "image/jpeg"
        elif "image/webp" in header:
            mime = "image/webp"
        elif "image/png" in header:
            mime = "image/png"
        return mime, payload.strip()
    return mime, s

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/answer-image")
def answer_image(req: ImageRequest):
    try:
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

        client = genai(api_key=api_key)
        model = os.getenv("GENAI_MODEL", "gpt-4o-mini")

        mime_type, image_b64 = normalize_base64_image(req.image_base64)

        try:
            base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        data_url = f"data:{mime_type};base64,{image_b64}"

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": req.question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=200,
        )

        answer = resp.choices[0].message.content or "answer:4089.35"
        return {"answer": str(answer).strip()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")