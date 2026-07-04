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

# Initialize FastAPI application instance
app = FastAPI(title="IITM Online Degree Curation Cell - Multimodal QA API")

# CRITICAL REQUIREMENT: Enable CORS globally for external evaluation suites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits access from any origin (e.g., Cloudflare Workers)
    allow_credentials=True,
    allow_methods=["*"],  # Permits all standard methods (GET, POST, etc.)
    allow_headers=["*"],  # Permits all custom headers
)

# Initialize the modern Google GenAI Client
# Ensure GEMINI_API_KEY is configured inside your cloud hosting platform variables

API_KEY = "AIzaSyYourActualKeyGoesInsideTheseQuotes"
client = genai.Client(api_key=API_KEY)


# --- API DATA SCHEMAS ---

class QARequest(BaseModel):
    image_base64: str
    question: str

class QAResponse(BaseModel):
    answer: str


# --- UTILITY CLEANING FILTERS ---

def clean_numeric_answer(text: str) -> str:
    """
    Enforces specification rule: For numeric answers, return only the number 
    without currency symbols, spaces, prefixes, commas, or extra text units.
    Example: '$4,089.35 USD' -> '4089.35'
    """
    text = text.strip()
    
    # Locate the first clustered collection of digits, periods, and commas
    numeric_match = re.search(r'([\d,]+\.?\d*)', text)
    if numeric_match:
        raw_num = numeric_match.group(1)
        # Remove commas commonly used as thousands separators
        cleaned_num = raw_num.replace(",", "")
        
        # Verify the filtered string evaluates strictly as a valid digit sequence
        if cleaned_num.replace(".", "", 1).isdigit():
            return cleaned_num
            
    return text


# --- CORE ROUTING ENDPOINTS ---

@app.post("/answer-image", response_model=QAResponse)
async def answer_image(payload: QARequest):
    """
    Multimodal QA Endpoint: Accepts a base64 encoded document image and a question.
    Extracts text/calculations using Gemini and handles output sanitisation.
    """
    # Guard against runtime errors if the API key environment setup was skipped
    if not API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Server Misconfiguration: GEMINI_API_KEY environment variable is missing."
        )

    # STEP 1: Safe Base64 Payload Stream Decoding
    try:
        # Strip structural data-URI metadata headers if provided by the client (e.g., "data:image/png;base64,")
        base64_clean = payload.image_base64.split(",")[-1]
        
        # Track and repair missing symmetric block padding constraints dynamically
        missing_padding = len(base64_clean) % 4
        if missing_padding:
            base64_clean += '=' * (4 - missing_padding)

        # Decode directly into raw byte blocks and wrap inside a PIL Image object
        image_bytes = base64.b64decode(base64_clean)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as base64_err:
        raise HTTPException(
            status_code=400, 
            detail=f"Base64 processing error. Invalid image format: {str(base64_err)}"
        )

    # STEP 2: Multi-modal Vision Model Inference Execution
    try:
        # Explicit system instructions force the model to comply with formatting criteria at source
        sys_instruction = (
            "You are a precision administrative and academic document text extraction engine. "
            "Examine the chart, graph, list, receipt, or invoice closely and answer the prompt directly. "
            "Provide only the targeted data asset. If the solution is a final number, sum, price, or "
            "statistical percentage, output ONLY the clear mathematical digits. Do not write full sentences."
        )

        # Call the stable multimodal framework
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, payload.question],
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.0  # Forces deterministic, absolute consistency across checker passes
            )
        )
        
        raw_output = response.text or ""
        
        # STEP 3: Apply Secondary Clean Filters to enforce string/numeric schema constraints
        final_answer = clean_numeric_answer(raw_output)
        
        return QAResponse(answer=final_answer)

    except Exception as inference_err:
        # Expose descriptive logging to easily troubleshoot external cloud runtime failures
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Engine Processing Error: {str(inference_err)}"
        )


@app.get("/")
def read_root():
    """Simple health check endpoint to verify web service health."""
    return {
        "status": "Online", 
        "engine": "IITM Online Degree Curation Cell Multimodal Backend"
    }

    