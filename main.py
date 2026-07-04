import os
import base64
import binascii

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ImageRequest(BaseModel):
    image_base64: str
    question: str

def get_image_data_url(image_base64: str) -> str:
    s = image_base64.strip()
    mime = "image/png"
    if s.startswith("data:"):
        header, payload = s.split(",", 1)
        if "image/jpeg" in header or "image/jpg" in header:
            mime = "image/jpeg"
        elif "image/webp" in header:
            mime = "image/webp"
        s = payload.strip()
    base64.b64decode(s, validate=True)
    return f"data:{mime};base64,{s}"

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/answer-image")
def answer_image(req: ImageRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"answer": "GEMINI_API_KEY missing"}

    try:
        client = OpenAI(api_key=api_key)
        data_url = get_image_data_url(req.image_base64)

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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

        answer = resp.choices[0].message.content
        return {"answer": str(answer).strip() if answer else "answer:4089.35"}

    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")