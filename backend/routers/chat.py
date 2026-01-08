
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import spacy
import google.generativeai as genai
import os
import traceback

router = APIRouter()

# Initialize Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# Initialize Gemini
def get_gemini_model():
    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-3-flash-preview')
    except Exception as e:
        print(f"Chat Router Error configuring Gemini: {e}")
        return None

class ChatInput(BaseModel):
    message: str

@router.post("/chat/secure")
async def secure_chat(payload: ChatInput):
    if not nlp:
        raise HTTPException(status_code=500, detail="Privacy layer not initialized (Spacy missing)")
    
    model = get_gemini_model()
    if not model:
        # Check env var specifically to give better error
        if not os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"):
             raise HTTPException(status_code=500, detail="LLM connection failed. API Key not found in environment.")
        raise HTTPException(status_code=500, detail="LLM connection failed (Gemini Config Error)")

    original_text = payload.message
    try:
        doc = nlp(original_text)
        
        # Redaction Logic
        redacted_text = original_text
        pii_map = {}
        
        sensitive_labels = ["PERSON", "GPE", "LOC", "ORG", "PHONE", "EMAIL"]
        counts = {}
        entities = list(doc.ents)
        replacements = []
        
        for ent in entities:
            if ent.label_ in sensitive_labels:
                label = ent.label_
                counts[label] = counts.get(label, 0) + 1
                token = f"<{label}_{counts[label]}>"
                pii_map[token] = ent.text
                replacements.append((ent.start_char, ent.end_char, token))
                
        # Apply replacements in reverse order
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        for start, end, token in replacements:
            redacted_text = redacted_text[:start] + token + redacted_text[end:]
            
        # Call Gemini
        response = model.generate_content(redacted_text)
        llm_response = response.text

        # Detokenization 
        restored_response = llm_response
        for token, original in pii_map.items():
            restored_response = restored_response.replace(token, original)

        return {
            "original_prompt": original_text,
            "redacted_prompt": redacted_text,
            "llm_response_raw": llm_response,
            "llm_response_restored": restored_response,
            "pii_map": pii_map
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Processing Error: {str(e)}", "redacted_prompt": locals().get("redacted_text", "N/A")}
