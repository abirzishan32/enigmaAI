
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import spacy
import google.generativeai as genai
import os
import traceback
from faker import Faker

router = APIRouter()
fake = Faker()

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
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"Chat Router Error configuring Gemini: {e}")
        return None

class ChatInput(BaseModel):
    message: str

def get_fake_pii(label):
    """Generate realistic fake data based on entity label"""
    if label == "PERSON":
        return fake.first_name()
    elif label == "GPE" or label == "LOC":
        return fake.city()
    elif label == "ORG":
        return fake.company()
    elif label == "EMAIL":
        return fake.email()
    elif label == "PHONE":
        return fake.phone_number()
    return f"[{label}]"

@router.post("/chat/secure")
async def secure_chat(payload: ChatInput):
    if not nlp:
        raise HTTPException(status_code=500, detail="Privacy layer not initialized (Spacy missing)")
    
    model = get_gemini_model()
    if not model:
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
        entities = list(doc.ents)
        replacements = []
        
        # Maintain consistency for the same entity text in this session
        session_map = {} 
        
        for ent in entities:
            if ent.label_ in sensitive_labels:
                if ent.text in session_map:
                    fake_val = session_map[ent.text]
                else:
                    fake_val = get_fake_pii(ent.label_)
                    session_map[ent.text] = fake_val
                
                # We map the fake value BACK to the original for restoration
                # But wait, if we have multiple "Alice" -> "Janice", we need to map "Janice" -> "Alice"
                # But what if "Bob" also maps to "Janice"? (Unlikely with Faker but possible)
                # For this simple implementation, we assume collision is rare enough.
                
                pii_map[fake_val] = ent.text
                replacements.append((ent.start_char, ent.end_char, fake_val))
                
        # Apply replacements in reverse order
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        for start, end, val in replacements:
            redacted_text = redacted_text[:start] + val + redacted_text[end:]
            
        # Call Gemini
        response = model.generate_content(redacted_text)
        llm_response = response.text

        # Detokenization 
        restored_response = llm_response
        for fake_val, original in pii_map.items():
            restored_response = restored_response.replace(fake_val, original)

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
