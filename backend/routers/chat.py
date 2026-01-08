
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

def is_personal_context(doc, ent):
    """
    Determine if an entity is likely strictly personal (needs redaction).
    Strict Heuristic:
    1. Direct Ownership: "My [Entity]"
    2. Personal State Verbs: Subject "I" + Verb "live/am/born/etc"
    
    If it's just a general action verb ("visit", "go"), we default to PRESERVE.
    """
    
    # 1. Check for Direct Ownership ("My name", "My city")
    # Traverse left children of the entity's root
    for child in ent.root.children:
        if child.dep_ == "poss" and child.text.lower() in ["my", "our"]:
            return True
    
    # Check if header of entity has ownership ("My name is Alice")
    # Entity "Alice" -> Head "is" -> Subject "name" -> Child "My"
    head = ent.root.head
    if head.pos_ == "NOUN":
        for child in head.children:
             if child.dep_ == "poss" and child.text.lower() in ["my", "our"]:
                 return True

    # 2. Check Verb-Subject Relation
    # Find the main verb
    verb = head
    while verb.pos_ != "VERB" and verb.pos_ != "AUX" and verb.head != verb:
        verb = verb.head
        
    PERSONAL_STATE_VERBS = {"live", "am", "born", "work", "study", "moved", "stay", "reside", "from"}
    
    # Special Handle for "be" verbs (am, is, was) mapping to lemma "be"
    # But checking raw text is safer for "am"
    
    if verb.pos_ in ["VERB", "AUX"]:
        # Check if subject is 1st person
        has_first_person_subject = False
        for child in verb.children:
            if child.dep_ in ["nsubj", "nsubjpass"] and child.text.lower() in ["i", "we", "me", "us"]:
                has_first_person_subject = True
                break
        
        # If subject is I/We, check if the verb is a "Personal State" verb
        if has_first_person_subject:
            # Check lemma or text
            if verb.lemma_.lower() in ["be", "live", "work", "study", "stay", "reside", "born", "move"]:
                 return True
            # Also check if it's "am" specifically
            if verb.text.lower() == "am":
                return True
                
    return False

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
        preserved_items = {} # Map of text -> label
        global_preserved_texts = set()

        # Phase 1: Identify Global Erasure Exceptions (Preservation)
        # If an entity is used in a non-personal (query) context ANYWHERE in the message,
        # we preserve it EVERYWHERE to maintain consistency for the LLM.
        for ent in entities:
             if ent.label_ in ["GPE", "LOC", "ORG"]:
                 # Check if this specific instance is strictly personal
                 is_personal = is_personal_context(doc, ent)
                 if not is_personal:
                     global_preserved_texts.add(ent.text)

        # Maintain consistency for the same entity text in this session
        session_map = {} 
        
        for ent in entities:
            if ent.label_ in sensitive_labels:
                
                # Check Global Preservation
                if ent.text in global_preserved_texts:
                    preserved_items[ent.text] = ent.label_
                    continue
                
                # Proceed to Redact
                if ent.text in session_map:
                    fake_val = session_map[ent.text]
                else:
                    fake_val = get_fake_pii(ent.label_)
                    session_map[ent.text] = fake_val
                
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
            "llm_response_restored": restored_response,
            "pii_map": pii_map,
            "preserved_items": preserved_items
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Processing Error: {str(e)}", "redacted_prompt": locals().get("redacted_text", "N/A")}
