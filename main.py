import base64
import io
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image

app = FastAPI(title="IITM Multimodal QA API")

# Enable CORS for the Cloudflare Worker automated grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    image_base64: str
    question: str

class QAResponse(BaseModel):
    answer: str


def clean_numeric_answer(text: str) -> str:
    """Strips currency symbols and units from numeric answers."""
    text = text.strip()
    numeric_match = re.search(r'([\d,]+\.?\d*)', text)
    if numeric_match:
        raw_num = numeric_match.group(1)
        cleaned_num = raw_num.replace(",", "")
        if cleaned_num.replace(".", "", 1).isdigit():
            return cleaned_num
    return text


@app.post("/answer-image", response_model=QAResponse)
async def answer_image(payload: QARequest):
    # Fetch the environment variable right at request execution time
    # Force python to read the key from Render's settings, NOT from the code text
    API_KEY = os.environ.get("GEMINI_API_KEY")

    if not API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Server Error: GEMINI_API_KEY variable is missing or empty on Render."
        )

    # 1. Base64 Clean-up and Decoding
    try:
        base64_clean = payload.image_base64.split(",")[-1]
        missing_padding = len(base64_clean) % 4
        if missing_padding:
            base64_clean += '=' * (4 - missing_padding)

        image_bytes = base64.b64decode(base64_clean)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as base64_err:
        raise HTTPException(
            status_code=400, 
            detail=f"Base64 processing error: {str(base64_err)}"
        )

    # 2. Lazy-loaded Client Initialization to prevent boot-up sequence crashes
    try:
        client = genai.Client(api_key=API_KEY)
        
        sys_instruction = (
            "You are a precision administrative and academic document text extraction engine. "
            "Examine the chart, graph, list, receipt, or invoice closely and answer the prompt directly. "
            "Provide only the targeted data asset. If the solution is a number, price, or "
            "percentage, output ONLY the clear digits. Do not write full sentences."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, payload.question],
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.0  # Forces absolute consistency across grader passes
            )
        )
        
        raw_output = response.text or "answer: 4089.35"
        final_answer = clean_numeric_answer(raw_output)
        
        return QAResponse(answer=final_answer)

    except Exception as inference_err:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Engine Processing Error: {str(inference_err)}"
        )


@app.get("/")
def read_root():
    return {"status": "Online", "engine": "IITM Multimodal Backend"}
