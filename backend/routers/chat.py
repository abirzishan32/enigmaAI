from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import google.generativeai as genai
import os
import traceback
import re
import math
from faker import Faker
from transformers import pipeline
from collections import defaultdict

router = APIRouter()
fake = Faker()

# ================================================================================
# MODULE 1: MULTI-MODEL NER PIPELINE
# ================================================================================

class NERPipeline:
    """
    Multi-model Named Entity Recognition combining:
    - BERT-NER (dslim/bert-base-NER) for primary entity extraction
    - Custom regex rules for pattern-based PII (emails, phones, SSN, etc.)
    - Context-aware entity boundary correction
    """
    
    def __init__(self):
        self.bert_ner = None
        self.label_map = {
            "PER": "PERSON",
            "ORG": "ORG", 
            "LOC": "LOC",
            "MISC": "MISC",
            "EMAIL": "EMAIL",
            "PHONE": "PHONE",
            "SSN": "SSN",
            "CREDIT_CARD": "CREDIT_CARD",
            "DATE_OF_BIRTH": "DATE_OF_BIRTH",
            "ADDRESS": "ADDRESS"
        }
        self._init_bert()
        
    def _init_bert(self):
        """Initialize BERT NER model"""
        try:
            print("🔄 Loading BERT NER model (dslim/bert-base-NER)...")
            self.bert_ner = pipeline(
                "ner", 
                model="dslim/bert-base-NER", 
                aggregation_strategy="simple"
            )
            print("✅ BERT NER model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
            self.bert_ner = None
    
    def _extract_regex_entities(self, text: str) -> List[Dict]:
        """Extract pattern-based PII using regex"""
        entities = []
        
        patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "DATE_OF_BIRTH": r'\b(?:born\s+(?:on\s+)?)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            "ADDRESS": r'\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln)\b'
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "entity_group": label,
                    "word": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,  # Regex matches are deterministic
                    "source": "regex"
                })
        
        return entities
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract all entities using multi-model approach.
        Returns unified entity list with normalized labels.
        """
        all_entities = []
        
        # 1. BERT NER extraction
        if self.bert_ner:
            bert_entities = self.bert_ner(text)
            for ent in bert_entities:
                ent["source"] = "bert"
                ent["entity_group"] = self.label_map.get(
                    ent["entity_group"], 
                    ent["entity_group"]
                )
                all_entities.append(ent)
        
        # 2. Regex-based extraction
        regex_entities = self._extract_regex_entities(text)
        
        # 3. Merge and deduplicate (prefer BERT for overlapping spans)
        all_entities = self._merge_entities(all_entities, regex_entities)
        
        return all_entities
    
    def _merge_entities(self, bert_ents: List[Dict], regex_ents: List[Dict]) -> List[Dict]:
        """Merge entities, handling overlaps (BERT takes precedence)"""
        merged = bert_ents.copy()
        
        for regex_ent in regex_ents:
            is_overlapping = False
            for bert_ent in bert_ents:
                # Check for overlap
                if not (regex_ent["end"] <= bert_ent["start"] or 
                        regex_ent["start"] >= bert_ent["end"]):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                merged.append(regex_ent)
        
        return sorted(merged, key=lambda x: x["start"])


# ================================================================================
# MODULE 2: INTENT CLASSIFICATION & CONTEXT ANALYSIS
# ================================================================================

class IntentType(Enum):
    """Classification of user intent regarding privacy"""
    QUERY = "query"           # "Tell me about Paris" - information seeking
    DISCLOSURE = "disclosure"  # "My name is Alice" - personal revelation
    HYBRID = "hybrid"          # "I live in Dhaka, recommend restaurants"


@dataclass
class IntentAnalysis:
    """Result of intent classification"""
    intent_type: IntentType
    confidence: float
    query_segments: List[Tuple[int, int]] = field(default_factory=list)
    disclosure_segments: List[Tuple[int, int]] = field(default_factory=list)
    query_keywords: List[str] = field(default_factory=list)
    disclosure_keywords: List[str] = field(default_factory=list)


class IntentClassifier:
    """
    Classifies user intent as QUERY, DISCLOSURE, or HYBRID.
    Uses pattern matching and contextual analysis.
    """
    
    # Query indicators - information seeking
    QUERY_PATTERNS = [
        r'\b(tell\s+me|explain|describe|what\s+is|who\s+is|where\s+is|how\s+(?:is|does|do|can|to))\b',
        r'\b(information\s+about|learn\s+about|know\s+about|read\s+about)\b',
        r'\b(recommend|suggest|find|search|look\s+up|show\s+me)\b',
        r'\b(compare|difference\s+between|versus|vs)\b',
        r'\b(history\s+of|facts\s+about|details\s+(?:of|about))\b',
        r'\?([\s]*$)',  # Question mark at end
        r'\b(can\s+you|could\s+you|would\s+you|please)\b'
    ]
    
    # Disclosure indicators - personal information sharing
    DISCLOSURE_PATTERNS = [
        r'\b(my\s+name\s+is|i\s+am\s+called|call\s+me)\b',
        r'\b(i\s+live|i\s+stay|i\s+reside|i\s+(?:am\s+)?from|i\s+was\s+born)\b',
        r'\b(i\s+work\s+(?:at|for)|i\s+am\s+employed|my\s+job|my\s+company)\b',
        r'\b(my\s+(?:email|phone|address|number|ssn|birthday))\b',
        r'\b(i\s+(?:am|\'m)\s+\d+\s+years?\s+old)\b',
        r'\b(my\s+(?:wife|husband|son|daughter|mother|father|brother|sister))\b',
        r'\b(contact\s+me\s+at|reach\s+me\s+at|my\s+contact)\b'
    ]
    
    def classify(self, text: str) -> IntentAnalysis:
        """Classify the intent of the input text"""
        text_lower = text.lower()
        
        query_score = 0.0
        disclosure_score = 0.0
        query_segments = []
        disclosure_segments = []
        query_keywords = []
        disclosure_keywords = []
        
        # Score query patterns
        for pattern in self.QUERY_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                query_score += 1.0
                query_segments.append((match.start(), match.end()))
                query_keywords.append(match.group())
        
        # Score disclosure patterns
        for pattern in self.DISCLOSURE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                disclosure_score += 1.5  # Weight disclosure higher (privacy sensitive)
                disclosure_segments.append((match.start(), match.end()))
                disclosure_keywords.append(match.group())
        
        # Normalize scores
        total_score = query_score + disclosure_score + 0.001  # Avoid division by zero
        query_confidence = query_score / total_score
        disclosure_confidence = disclosure_score / total_score
        
        # Determine intent type
        if disclosure_score > 0 and query_score > 0:
            intent_type = IntentType.HYBRID
            confidence = min(query_confidence, disclosure_confidence) * 2  # Confidence in hybrid
        elif disclosure_score > query_score:
            intent_type = IntentType.DISCLOSURE
            confidence = disclosure_confidence
        else:
            intent_type = IntentType.QUERY
            confidence = max(query_confidence, 0.5)  # Default to query with moderate confidence
        
        return IntentAnalysis(
            intent_type=intent_type,
            confidence=confidence,
            query_segments=query_segments,
            disclosure_segments=disclosure_segments,
            query_keywords=query_keywords,
            disclosure_keywords=disclosure_keywords
        )


# ================================================================================
# MODULE 3: ENTITY SENSITIVITY SCORING
# ================================================================================

@dataclass
class SensitivityScore:
    """Comprehensive sensitivity assessment for an entity"""
    entity_text: str
    entity_type: str
    identity_risk: float      # 0-1: How much does this identify the person?
    query_necessity: float    # 0-1: How important is this for the query?
    reidentification_risk: float  # 0-1: k-anonymity based risk
    combined_score: float     # Weighted combination
    recommended_strategy: str  # A, B, C, or D


class SensitivityScorer:
    """
    Computes multi-dimensional sensitivity scores for entities.
    
    Dimensions:
    1. Identity Risk - How uniquely identifying is this entity?
    2. Query Necessity - Is this entity required to answer the query?
    3. Re-identification Risk - Based on quasi-identifier combinations
    """
    
    # Base identity risk by entity type
    IDENTITY_RISK_BASE = {
        "PERSON": 0.95,      # Names are highly identifying
        "EMAIL": 0.99,       # Emails are unique identifiers
        "PHONE": 0.95,       # Phone numbers are highly identifying
        "SSN": 1.0,          # SSN is a direct identifier
        "CREDIT_CARD": 0.98, # Financial identifiers
        "ADDRESS": 0.85,     # Addresses can be identifying
        "DATE_OF_BIRTH": 0.7, # DOB + other info is identifying
        "LOC": 0.3,          # Locations are generally public
        "ORG": 0.4,          # Organizations are generally public
        "MISC": 0.2          # Miscellaneous - context dependent
    }
    
    # Population-based quasi-identifier weights (for re-identification)
    QUASI_IDENTIFIER_WEIGHTS = {
        "LOC": 0.4,           # Location narrows population
        "ORG": 0.3,           # Employer narrows population
        "DATE_OF_BIRTH": 0.5, # Age/DOB narrows significantly
        "MISC": 0.1
    }
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
    
    def _compute_identity_risk(self, entity: Dict, context: str) -> float:
        """
        Compute identity risk score.
        Higher = more identifying, needs redaction.
        """
        entity_type = entity["entity_group"]
        base_risk = self.IDENTITY_RISK_BASE.get(entity_type, 0.5)
        
        # Contextual modifiers
        text_lower = context.lower()
        entity_start = entity["start"]
        
        # Check for possessive context ("my", "our")
        before_context = text_lower[max(0, entity_start-20):entity_start]
        if re.search(r'\b(my|our)\s*$', before_context):
            base_risk = min(1.0, base_risk + 0.3)
        
        # Check for first-person state verbs
        if re.search(r'\b(i|we)\s+(live|work|am|was|born)\b', before_context):
            base_risk = min(1.0, base_risk + 0.25)
        
        return base_risk
    
    def _compute_query_necessity(self, entity: Dict, intent: IntentAnalysis, context: str) -> float:
        """
        Compute how necessary this entity is for answering the query.
        Higher = more important to preserve.
        """
        entity_text = entity["word"].lower()
        entity_start = entity["start"]
        entity_end = entity["end"]
        
        # If pure disclosure, entity is not query-necessary
        if intent.intent_type == IntentType.DISCLOSURE:
            return 0.1
        
        # Check if entity is in query segments
        for q_start, q_end in intent.query_segments:
            # Entity is near or within query keywords
            if abs(entity_start - q_end) < 30 or abs(entity_end - q_start) < 30:
                return 0.9
        
        # Check for query-related patterns around entity
        context_lower = context.lower()
        before = context_lower[max(0, entity_start-50):entity_start]
        after = context_lower[entity_end:entity_end+30]
        
        query_indicators = [
            r'(about|regarding|concerning)\s*$',
            r'(visit|go\s+to|travel\s+to)\s*$',
            r'(in|at|near)\s*$',
            r'^(\s*\?)',
            r'(recommend|suggest|find)\s+\w*\s*$'
        ]
        
        for pattern in query_indicators:
            if re.search(pattern, before) or re.search(pattern, after):
                return 0.85
        
        # Default moderate necessity for QUERY intent
        if intent.intent_type == IntentType.QUERY:
            return 0.6
        
        # HYBRID: moderate
        return 0.4
    
    def _compute_reidentification_risk(self, entity: Dict, all_entities: List[Dict]) -> float:
        """
        Compute re-identification risk based on quasi-identifier combinations.
        Uses simplified k-anonymity estimation.
        """
        entity_type = entity["entity_group"]
        
        # Count quasi-identifiers in the message
        quasi_identifiers = [e for e in all_entities 
                           if e["entity_group"] in self.QUASI_IDENTIFIER_WEIGHTS]
        
        if len(quasi_identifiers) == 0:
            return self.QUASI_IDENTIFIER_WEIGHTS.get(entity_type, 0.1)
        
        # More quasi-identifiers = higher re-identification risk
        # Simplified: risk increases with combination of quasi-identifiers
        combined_weight = sum(
            self.QUASI_IDENTIFIER_WEIGHTS.get(e["entity_group"], 0.1) 
            for e in quasi_identifiers
        )
        
        # Normalize to 0-1 (cap at 3 quasi-identifiers for max risk)
        risk = min(1.0, combined_weight / 1.5)
        
        return risk
    
    def score_entity(self, entity: Dict, all_entities: List[Dict], 
                    intent: IntentAnalysis, context: str) -> SensitivityScore:
        """Compute comprehensive sensitivity score for an entity"""
        
        identity_risk = self._compute_identity_risk(entity, context)
        query_necessity = self._compute_query_necessity(entity, intent, context)
        reidentification_risk = self._compute_reidentification_risk(entity, all_entities)
        
        # Combined score: weighted average
        # Higher identity risk and reidentification = needs protection
        # Higher query necessity = needs preservation
        # Formula: protection_need = (identity + reidentification) / 2 - query_necessity * 0.5
        
        protection_need = (identity_risk * 0.5 + reidentification_risk * 0.3) - (query_necessity * 0.3)
        combined_score = max(0, min(1, protection_need + 0.3))  # Bias towards protection
        
        # Determine recommended strategy
        if identity_risk >= 0.8 or entity["entity_group"] in ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]:
            strategy = "A"  # Full Redaction
        elif query_necessity >= 0.7 and identity_risk < 0.5:
            strategy = "C"  # Preservation
        elif entity["entity_group"] in ["LOC"] and identity_risk >= 0.5:
            strategy = "D"  # Geographic Obfuscation
        else:
            strategy = "B"  # Generalization
        
        return SensitivityScore(
            entity_text=entity["word"],
            entity_type=entity["entity_group"],
            identity_risk=round(identity_risk, 3),
            query_necessity=round(query_necessity, 3),
            reidentification_risk=round(reidentification_risk, 3),
            combined_score=round(combined_score, 3),
            recommended_strategy=strategy
        )


# ================================================================================
# MODULE 4: ADAPTIVE PRIVACY STRATEGY SELECTOR
# ================================================================================

class PrivacyStrategy(Enum):
    """Privacy transformation strategies"""
    FULL_REDACTION = "A"       # Complete replacement with synthetic data
    GENERALIZATION = "B"       # Hierarchical generalization (Alice → Female Name)
    PRESERVATION = "C"         # Keep original (query-critical, low risk)
    GEO_OBFUSCATION = "D"      # Geographic clustering (Dhaka → South Asian City)


@dataclass
class TransformationResult:
    """Result of applying a privacy strategy"""
    original: str
    transformed: str
    strategy: PrivacyStrategy
    entity_type: str
    is_reversible: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrivacyStrategySelector:
    """
    Selects and applies appropriate privacy transformation strategy.
    Considers entity sensitivity scores and intent context.
    """
    
    def __init__(self):
        self.faker = Faker()
        self.transformer = SemanticTransformer()
    
    def select_strategy(self, score: SensitivityScore, intent: IntentAnalysis) -> PrivacyStrategy:
        """Select optimal privacy strategy based on scores and intent"""
        
        # Override: Direct identifiers always get full redaction
        if score.entity_type in ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "ADDRESS"]:
            return PrivacyStrategy.FULL_REDACTION
        
        # Strategy selection based on recommended strategy from scorer
        strategy_map = {
            "A": PrivacyStrategy.FULL_REDACTION,
            "B": PrivacyStrategy.GENERALIZATION,
            "C": PrivacyStrategy.PRESERVATION,
            "D": PrivacyStrategy.GEO_OBFUSCATION
        }
        
        base_strategy = strategy_map.get(score.recommended_strategy, PrivacyStrategy.GENERALIZATION)
        
        # Intent-based adjustments
        if intent.intent_type == IntentType.QUERY and score.query_necessity > 0.7:
            # Preserve query-critical entities
            if score.identity_risk < 0.6:
                return PrivacyStrategy.PRESERVATION
        
        if intent.intent_type == IntentType.DISCLOSURE:
            # More aggressive protection for disclosures
            if base_strategy == PrivacyStrategy.PRESERVATION:
                return PrivacyStrategy.GENERALIZATION
        
        return base_strategy
    
    def apply_strategy(self, entity: Dict, strategy: PrivacyStrategy, 
                      score: SensitivityScore) -> TransformationResult:
        """Apply the selected privacy strategy to an entity"""
        
        original = entity["word"]
        entity_type = entity["entity_group"]
        
        if strategy == PrivacyStrategy.FULL_REDACTION:
            transformed = self.transformer.full_redaction(original, entity_type)
            is_reversible = True
            
        elif strategy == PrivacyStrategy.GENERALIZATION:
            transformed = self.transformer.generalize(original, entity_type)
            is_reversible = False  # Information loss
            
        elif strategy == PrivacyStrategy.GEO_OBFUSCATION:
            transformed = self.transformer.geo_obfuscate(original)
            is_reversible = False
            
        else:  # PRESERVATION
            transformed = original
            is_reversible = True
        
        return TransformationResult(
            original=original,
            transformed=transformed,
            strategy=strategy,
            entity_type=entity_type,
            is_reversible=is_reversible,
            metadata={
                "identity_risk": score.identity_risk,
                "query_necessity": score.query_necessity,
                "combined_score": score.combined_score
            }
        )


# ================================================================================
# MODULE 5: SEMANTIC-PRESERVING TRANSFORMATION
# ================================================================================

class SemanticTransformer:
    """
    Applies semantic-preserving transformations to entities.
    Maintains contextual meaning while protecting identity.
    """
    
    def __init__(self):
        self.faker = Faker()
        
        # Geographic clustering mappings
        self.geo_clusters = {
            # South Asia
            "dhaka": "South Asian City", "mumbai": "South Asian City",
            "delhi": "South Asian City", "kolkata": "South Asian City",
            "chennai": "South Asian City", "bangalore": "South Asian City",
            "karachi": "South Asian City", "lahore": "South Asian City",
            "colombo": "South Asian City", "kathmandu": "South Asian City",
            
            # East Asia
            "tokyo": "East Asian City", "beijing": "East Asian City",
            "shanghai": "East Asian City", "seoul": "East Asian City",
            "hong kong": "East Asian City", "taipei": "East Asian City",
            
            # Southeast Asia
            "singapore": "Southeast Asian City", "bangkok": "Southeast Asian City",
            "jakarta": "Southeast Asian City", "manila": "Southeast Asian City",
            "kuala lumpur": "Southeast Asian City", "hanoi": "Southeast Asian City",
            
            # Europe
            "london": "European City", "paris": "European City",
            "berlin": "European City", "rome": "European City",
            "madrid": "European City", "amsterdam": "European City",
            "vienna": "European City", "prague": "European City",
            
            # North America
            "new york": "North American City", "los angeles": "North American City",
            "chicago": "North American City", "toronto": "North American City",
            "san francisco": "North American City", "boston": "North American City",
            "seattle": "North American City", "miami": "North American City",
            
            # Middle East
            "dubai": "Middle Eastern City", "abu dhabi": "Middle Eastern City",
            "riyadh": "Middle Eastern City", "doha": "Middle Eastern City",
            
            # Australia/Oceania
            "sydney": "Oceanian City", "melbourne": "Oceanian City",
            "auckland": "Oceanian City", "brisbane": "Oceanian City",
            
            # Africa
            "cairo": "African City", "lagos": "African City",
            "johannesburg": "African City", "nairobi": "African City",
            
            # South America
            "sao paulo": "South American City", "buenos aires": "South American City",
            "rio de janeiro": "South American City", "lima": "South American City"
        }
        
        # Hierarchical name generalizations
        self.name_generalizations = {
            "male": ["a person", "someone", "an individual", "a man"],
            "female": ["a person", "someone", "an individual", "a woman"],
            "neutral": ["a person", "someone", "an individual"]
        }
        
        # Common first names for gender inference (simplified)
        self.male_names = {"john", "james", "robert", "michael", "william", "david", 
                          "richard", "joseph", "thomas", "charles", "ahmed", "mohammad",
                          "raj", "amit", "rahul", "vikram", "arjun", "sanjay"}
        self.female_names = {"mary", "patricia", "jennifer", "linda", "elizabeth",
                            "barbara", "susan", "jessica", "sarah", "karen", "fatima",
                            "priya", "anita", "sunita", "rekha", "meena", "kavita"}
    
    def full_redaction(self, text: str, entity_type: str) -> str:
        """Complete replacement with realistic synthetic data"""
        
        if entity_type == "PERSON":
            return self.faker.first_name()
        elif entity_type in ["LOC", "GPE"]:
            return self.faker.city()
        elif entity_type == "ORG":
            return self.faker.company()
        elif entity_type == "EMAIL":
            return self.faker.email()
        elif entity_type == "PHONE":
            return self.faker.phone_number()
        elif entity_type == "SSN":
            return "XXX-XX-XXXX"
        elif entity_type == "CREDIT_CARD":
            return "XXXX-XXXX-XXXX-XXXX"
        elif entity_type == "ADDRESS":
            return self.faker.street_address()
        elif entity_type == "DATE_OF_BIRTH":
            return "[DATE]"
        else:
            return f"[{entity_type}]"
    
    def generalize(self, text: str, entity_type: str) -> str:
        """Hierarchical generalization preserving semantic category"""
        
        if entity_type == "PERSON":
            # Infer gender and generalize
            text_lower = text.lower().split()[0] if text else ""
            if text_lower in self.male_names:
                return "a person"
            elif text_lower in self.female_names:
                return "a person"
            return "someone"
        
        elif entity_type in ["LOC", "GPE"]:
            # Try geographic clustering first
            text_lower = text.lower()
            if text_lower in self.geo_clusters:
                return self.geo_clusters[text_lower]
            return "a city"
        
        elif entity_type == "ORG":
            return "an organization"
        
        elif entity_type == "EMAIL":
            return "[email address]"
        
        elif entity_type == "PHONE":
            return "[phone number]"
        
        elif entity_type == "DATE_OF_BIRTH":
            return "[a date]"
        
        return f"[{entity_type.lower()}]"
    
    def geo_obfuscate(self, location: str) -> str:
        """Geographic clustering - map to regional category"""
        location_lower = location.lower()
        
        # Direct lookup
        if location_lower in self.geo_clusters:
            return self.geo_clusters[location_lower]
        
        # Fuzzy matching for partial matches
        for city, cluster in self.geo_clusters.items():
            if city in location_lower or location_lower in city:
                return cluster
        
        # Default: generic
        return "a city"
    
    def contextual_replacement(self, text: str, entity_type: str, context: str) -> str:
        """Context-aware replacement with similar entities"""
        # For now, falls back to full redaction
        # Future: Use embeddings to find contextually similar replacements
        return self.full_redaction(text, entity_type)


# ================================================================================
# MODULE 6: RESPONSE POST-PROCESSING & EVALUATION
# ================================================================================

@dataclass
class PrivacyMetrics:
    """Privacy-utility tradeoff metrics"""
    entities_detected: int
    entities_redacted: int
    entities_generalized: int
    entities_preserved: int
    entities_geo_obfuscated: int
    privacy_score: float        # 0-1: How well is privacy protected?
    utility_score: float        # 0-1: How much semantic meaning is preserved?
    tradeoff_score: float       # Combined metric


class ResponseProcessor:
    """
    Post-processes LLM responses and computes evaluation metrics.
    Handles de-anonymization for reversible transformations.
    """
    
    def __init__(self):
        pass
    
    def restore_response(self, response: str, pii_map: Dict[str, str]) -> str:
        """Restore original entities in the response (de-anonymization)"""
        restored = response
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_map = sorted(pii_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for fake_val, original in sorted_map:
            restored = restored.replace(fake_val, original)
        
        return restored
    
    def compute_metrics(self, transformations: List[TransformationResult]) -> PrivacyMetrics:
        """Compute privacy-utility tradeoff metrics"""
        
        if not transformations:
            return PrivacyMetrics(
                entities_detected=0, entities_redacted=0, entities_generalized=0,
                entities_preserved=0, entities_geo_obfuscated=0,
                privacy_score=1.0, utility_score=1.0, tradeoff_score=1.0
            )
        
        # Count by strategy
        counts = defaultdict(int)
        for t in transformations:
            counts[t.strategy] += 1
        
        total = len(transformations)
        redacted = counts[PrivacyStrategy.FULL_REDACTION]
        generalized = counts[PrivacyStrategy.GENERALIZATION]
        preserved = counts[PrivacyStrategy.PRESERVATION]
        geo_obfuscated = counts[PrivacyStrategy.GEO_OBFUSCATION]
        
        # Privacy score: higher when more entities are protected
        # Redaction = 1.0, Generalization = 0.8, GeoObfuscation = 0.7, Preservation = 0.0
        privacy_score = (
            redacted * 1.0 + 
            generalized * 0.8 + 
            geo_obfuscated * 0.7 + 
            preserved * 0.0
        ) / total if total > 0 else 1.0
        
        # Utility score: higher when semantic meaning is preserved
        # Preservation = 1.0, GeoObfuscation = 0.8, Generalization = 0.6, Redaction = 0.4
        utility_score = (
            preserved * 1.0 + 
            geo_obfuscated * 0.8 + 
            generalized * 0.6 + 
            redacted * 0.4
        ) / total if total > 0 else 1.0
        
        # Tradeoff: harmonic mean of privacy and utility
        if privacy_score + utility_score > 0:
            tradeoff_score = 2 * (privacy_score * utility_score) / (privacy_score + utility_score)
        else:
            tradeoff_score = 0.0
        
        return PrivacyMetrics(
            entities_detected=total,
            entities_redacted=redacted,
            entities_generalized=generalized,
            entities_preserved=preserved,
            entities_geo_obfuscated=geo_obfuscated,
            privacy_score=round(privacy_score, 3),
            utility_score=round(utility_score, 3),
            tradeoff_score=round(tradeoff_score, 3)
        )


# ================================================================================
# MAIN PIPELINE ORCHESTRATOR
# ================================================================================

class AdaptivePrivacyPipeline:
    """
    Main orchestrator for the adaptive privacy-preserving NLP pipeline.
    Coordinates all modules for end-to-end processing.
    """
    
    def __init__(self):
        self.ner_pipeline = NERPipeline()
        self.intent_classifier = IntentClassifier()
        self.sensitivity_scorer = SensitivityScorer()
        self.strategy_selector = PrivacyStrategySelector()
        self.response_processor = ResponseProcessor()
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process input text through the complete privacy pipeline.
        
        Returns comprehensive result including:
        - Original and transformed text
        - Entity analysis
        - Privacy metrics
        - Transformation details
        """
        
        # Step 1: Extract entities (Multi-Model NER)
        entities = self.ner_pipeline.extract_entities(text)
        
        # Step 2: Classify intent
        intent = self.intent_classifier.classify(text)
        
        # Step 3: Score each entity's sensitivity
        scored_entities = []
        for entity in entities:
            score = self.sensitivity_scorer.score_entity(
                entity, entities, intent, text
            )
            scored_entities.append((entity, score))
        
        # Step 4: Select and apply strategies
        transformations = []
        pii_map = {}  # fake -> original (for reversible transformations)
        session_map = {}  # original -> fake (for consistency)
        preserved_items = {}
        
        for entity, score in scored_entities:
            strategy = self.strategy_selector.select_strategy(score, intent)
            result = self.strategy_selector.apply_strategy(entity, strategy, score)
            transformations.append(result)
            
            if strategy == PrivacyStrategy.PRESERVATION:
                preserved_items[result.original] = score.entity_type
            elif result.is_reversible and result.transformed != result.original:
                # Maintain consistency
                if result.original in session_map:
                    result.transformed = session_map[result.original]
                else:
                    session_map[result.original] = result.transformed
                pii_map[result.transformed] = result.original
        
        # Step 5: Apply transformations to text (reverse order for correct indices)
        transformed_text = text
        replacements = []
        
        for i, (entity, score) in enumerate(scored_entities):
            result = transformations[i]
            if result.strategy != PrivacyStrategy.PRESERVATION:
                # Use consistent replacement from session_map
                replacement = session_map.get(result.original, result.transformed)
                replacements.append((entity["start"], entity["end"], replacement))
        
        # Sort by start position (reverse) and apply
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, replacement in replacements:
            transformed_text = transformed_text[:start] + replacement + transformed_text[end:]
        
        # Step 6: Compute metrics
        metrics = self.response_processor.compute_metrics(transformations)
        
        return {
            "original_text": text,
            "transformed_text": transformed_text,
            "intent": {
                "type": intent.intent_type.value,
                "confidence": round(intent.confidence, 3),
                "query_keywords": intent.query_keywords[:5],
                "disclosure_keywords": intent.disclosure_keywords[:5]
            },
            "entities": [
                {
                    "text": e["word"],
                    "type": e["entity_group"],
                    "start": e["start"],
                    "end": e["end"],
                    "source": e.get("source", "bert"),
                    "sensitivity": {
                        "identity_risk": s.identity_risk,
                        "query_necessity": s.query_necessity,
                        "reidentification_risk": s.reidentification_risk,
                        "combined_score": s.combined_score,
                        "strategy": s.recommended_strategy
                    }
                }
                for e, s in scored_entities
            ],
            "transformations": [
                {
                    "original": t.original,
                    "transformed": t.transformed,
                    "strategy": t.strategy.value,
                    "entity_type": t.entity_type,
                    "reversible": t.is_reversible
                }
                for t in transformations
            ],
            "pii_map": pii_map,
            "preserved_items": preserved_items,
            "metrics": {
                "entities_detected": metrics.entities_detected,
                "entities_redacted": metrics.entities_redacted,
                "entities_generalized": metrics.entities_generalized,
                "entities_preserved": metrics.entities_preserved,
                "entities_geo_obfuscated": metrics.entities_geo_obfuscated,
                "privacy_score": metrics.privacy_score,
                "utility_score": metrics.utility_score,
                "tradeoff_score": metrics.tradeoff_score
            }
        }


# ================================================================================
# INITIALIZATION
# ================================================================================

# Initialize the pipeline
print("=" * 60)
print("🚀 Initializing Adaptive Privacy Pipeline v2.0")
print("=" * 60)

privacy_pipeline = AdaptivePrivacyPipeline()

def get_gemini_model():
    """Initialize Gemini LLM"""
    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"❌ Gemini configuration error: {e}")
        return None


# ================================================================================
# API ENDPOINTSs
# ================================================================================

class ChatInput(BaseModel):
    message: str


@router.post("/chat/secure")
async def secure_chat(payload: ChatInput):
    """
    Privacy-preserving chat endpoint.
    
    Processes user input through the adaptive privacy pipeline,
    sends sanitized prompt to Gemini, and restores the response.
    """
    
    if not privacy_pipeline.ner_pipeline.bert_ner:
        raise HTTPException(
            status_code=500, 
            detail="Privacy layer not initialized (BERT NER missing)"
        )
    
    model = get_gemini_model()
    if not model:
        if not os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="LLM connection failed. API Key not found in environment."
            )
        raise HTTPException(
            status_code=500, 
            detail="LLM connection failed (Gemini Config Error)"
        )

    original_text = payload.message
    
    try:
        # Process through privacy pipeline
        pipeline_result = privacy_pipeline.process(original_text)
        
        redacted_text = pipeline_result["transformed_text"]
        pii_map = pipeline_result["pii_map"]
        
        # Call Gemini with sanitized prompt
        response = model.generate_content(redacted_text)
        llm_response = response.text

        # Restore original entities in response
        restored_response = privacy_pipeline.response_processor.restore_response(
            llm_response, pii_map
        )

        return {
            "original_prompt": original_text,
            "redacted_prompt": redacted_text,
            "llm_response_raw": llm_response,
            "llm_response_restored": restored_response,
            "pii_map": pii_map,
            "preserved_items": pipeline_result["preserved_items"],
            "intent": pipeline_result["intent"],
            "entities": pipeline_result["entities"],
            "transformations": pipeline_result["transformations"],
            "metrics": pipeline_result["metrics"]
        }
        
    except Exception as e:
        traceback.print_exc()
        return {
            "error": f"Processing Error: {str(e)}", 
            "redacted_prompt": locals().get("redacted_text", "N/A")
        }


@router.post("/chat/analyze")
async def analyze_privacy(payload: ChatInput):
    """
    Analyze input without calling LLM.
    Useful for debugging and understanding privacy decisions.
    """
    
    if not privacy_pipeline.ner_pipeline.bert_ner:
        raise HTTPException(
            status_code=500, 
            detail="Privacy layer not initialized (BERT NER missing)"
        )
    
    try:
        result = privacy_pipeline.process(payload.message)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Analysis Error: {str(e)}"}
