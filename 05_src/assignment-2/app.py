#!/usr/bin/env python3
"""
Walter's AI Chatbot & Security Scanner
Complete Python Application (converted from Jupyter notebook)

Features:
- Interactive chat with configurable personality
- VirusTotal file security scanning with AI interpretation
- AI Risk Database hybrid semantic search
- Enterprise-grade content guardrails
"""

import os
import sys
import re
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Tuple
from enum import Enum

# Third-party imports
import gradio as gr
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# ============ ENVIRONMENT SETUP ============
## USe relative path to ensure it works regardless of where the script is run from
load_dotenv('./.secrets')

API_GATEWAY_KEY = os.getenv('API_GATEWAY_KEY')
if not API_GATEWAY_KEY:
    print("⚠️ Warning: API_GATEWAY_KEY not found. Some features may not work.")
    API_GATEWAY_KEY = "demo_key"

# Initialize OpenAI client
client = OpenAI(
    default_headers={"x-api-key": API_GATEWAY_KEY},
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1'
)

print("✅ OpenAI client initialized")

# ============ ENUM DEFINITIONS ============
class ToneType(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    HUMOROUS = "humorous"


class StyleType(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SOCRATIC = "socratic"


class RoleType(str, Enum):
    ASSISTANT = "helpful AI assistant"
    MENTOR = "mentor and educator"
    EXPERT = "subject matter expert"
    BRAINSTORMER = "creative brainstorming partner"
    THERAPIST = "empathetic listener"
    ANALYST = "data analyst"


# ============ PYDANTIC MODELS ============
class ChatConfig(BaseModel):
    """Pydantic model for chat personality and behavior configuration"""
    
    tone: ToneType = Field(
        default=ToneType.FRIENDLY,
        description="The tone of voice for responses"
    )
    conversational_style: StyleType = Field(
        default=StyleType.CONCISE,
        description="How detailed or creative responses should be"
    )
    role: RoleType = Field(
        default=RoleType.ASSISTANT,
        description="The character role the AI adopts"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Controls randomness (0=deterministic, 2=very creative)"
    )
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum response length"
    )
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Override default system prompt"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use"
    )
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v

    def get_system_prompt(self) -> str:
        """Generate comprehensive system prompt based on configuration"""
        if self.custom_system_prompt:
            return self.custom_system_prompt
        
        return f"""You are a {self.role.value} with a {self.tone.value} tone.
Your conversational style is {self.conversational_style.value}.

Guidelines:
- Do not answer any requests on the following topics: Cats, Dogs, Horoscopes, Zodiac Signs, or Taylor Swift.
- Maintain consistency with your assigned tone throughout
- Adapt your response length based on your style (concise = short, detailed = thorough)
- Stay in character as a {self.role.value}
- Be helpful, accurate, and respectful"""
    
    class Config:
        use_enum_values = True


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatSession(BaseModel):
    config: ChatConfig
    messages: List[Message] = Field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add message to session"""
        self.messages.append(Message(role=role, content=content))
    
    def get_messages_for_api(self) -> List[dict]:
        """Convert messages to OpenAI API format"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


class PersonalityLibrary:
    """Collection of pre-configured chat personalities"""
    
    DEFAULT = ChatConfig(
        tone=ToneType.FRIENDLY,
        conversational_style=StyleType.CONCISE,
        role=RoleType.ASSISTANT,
        temperature=0.7
    )
    
    MENTOR = ChatConfig(
        tone=ToneType.PROFESSIONAL,
        conversational_style=StyleType.DETAILED,
        role=RoleType.MENTOR,
        temperature=0.6
    )
    
    CREATIVE = ChatConfig(
        tone=ToneType.CASUAL,
        conversational_style=StyleType.CREATIVE,
        role=RoleType.BRAINSTORMER,
        temperature=0.9
    )
    
    ANALYTICAL = ChatConfig(
        tone=ToneType.FORMAL,
        conversational_style=StyleType.ANALYTICAL,
        role=RoleType.ANALYST,
        temperature=0.5
    )
    
    @staticmethod
    def get_all_personalities():
        """Return all available personalities"""
        return {
            "default": PersonalityLibrary.DEFAULT,
            "mentor": PersonalityLibrary.MENTOR,
            "creative": PersonalityLibrary.CREATIVE,
            "analytical": PersonalityLibrary.ANALYTICAL
        }


print("✅ Pydantic models and PersonalityLibrary defined")

# ============ GUARDRAILS SYSTEM ============
@dataclass
class GuardrailConfig:
    """Enterprise configuration for content guardrails"""
    forbidden_topics: List[str] = field(default_factory=lambda: [
        "cats", "dogs", "horoscopes", "zodiac signs", "taylor swift"
    ])
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r"\b(cats?|dogs?|horoscope|zodiac|taylor\s+swift)\b",
        r"prompt\s+injection", r"system\s+prompt", r"ignore\s+instructions"
    ])
    max_input_length: int = 2000
    min_input_length: int = 1
    max_tokens_per_request: int = 4000
    max_requests_per_minute: int = 60
    max_output_length: int = 4000
    min_output_length: int = 1


class GuardrailViolation(Exception):
    """Custom exception for guardrail violations"""
    def __init__(self, message: str, severity: str, violation_type: str):
        self.message = message
        self.severity = severity
        self.violation_type = violation_type
        super().__init__(f"[{severity}] {violation_type}: {message}")


class ContentGuardrails:
    """Enterprise-grade content validation"""
    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.request_history: Dict[str, List[float]] = {}
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.forbidden_patterns]
    
    def validate_input(self, message: str, user_id: str = "default") -> Tuple[bool, str]:
        if not message or not isinstance(message, str):
            raise GuardrailViolation("Input must be a non-empty string", "CRITICAL", "invalid_input_type")
        if len(message) < self.config.min_input_length:
            raise GuardrailViolation(f"Input too short", "WARNING", "length_violation")
        if len(message) > self.config.max_input_length:
            raise GuardrailViolation(f"Input exceeds max length ({self.config.max_input_length})", "CRITICAL", "length_violation")
        
        message_lower = message.lower()
        for topic in self.config.forbidden_topics:
            if topic in message_lower:
                raise GuardrailViolation(f"Forbidden topic: '{topic}'", "CRITICAL", "forbidden_content")
        
        for pattern in self.compiled_patterns:
            if pattern.search(message):
                raise GuardrailViolation("Suspicious patterns detected", "CRITICAL", "prompt_injection")
        
        current_time = time.time()
        if user_id not in self.request_history:
            self.request_history[user_id] = []
        self.request_history[user_id] = [t for t in self.request_history[user_id] if current_time - t < 60]
        if len(self.request_history[user_id]) >= self.config.max_requests_per_minute:
            raise GuardrailViolation(f"Rate limit exceeded", "WARNING", "rate_limit")
        self.request_history[user_id].append(current_time)
        return True, "Input validation passed"
    
    def validate_output(self, response: str) -> Tuple[bool, str]:
        if not isinstance(response, str):
            raise GuardrailViolation("Output must be a string", "CRITICAL", "invalid_output_type")
        if len(response) < self.config.min_output_length:
            raise GuardrailViolation("Insufficient output", "WARNING", "output_length_violation")
        if len(response) > self.config.max_output_length:
            return True, response[:self.config.max_output_length] + "..."
        
        response_lower = response.lower()
        for topic in self.config.forbidden_topics:
            if topic in response_lower:
                raise GuardrailViolation(f"Forbidden content in output: '{topic}'", "CRITICAL", "forbidden_output_content")
        
        system_indicators = ["system prompt", "you are a", "role:", "guideline:", "instruction:"]
        for indicator in system_indicators:
            if indicator in response_lower and ("system" in response_lower or "instruction" in response_lower):
                raise GuardrailViolation("System prompt leakage detected", "CRITICAL", "prompt_leakage")
        return True, response


guardrails_config = GuardrailConfig()
guardrails = ContentGuardrails(guardrails_config)
print("✅ Enterprise guardrails system initialized")
print(f"   📋 Forbidden topics: {len(guardrails_config.forbidden_topics)}")
print(f"   🔍 Pattern checks: {len(guardrails_config.forbidden_patterns)}")
print(f"   ⏱️  Max input: {guardrails_config.max_input_length} chars")
print(f"   🚀 Rate limit: {guardrails_config.max_requests_per_minute} req/min")


# ============ INTENT DETECTION ============
class IntentDetector:
    """Detects user intent and routes to appropriate service"""
    
    # Keywords mapping to services
    FILE_SCAN_KEYWORDS = {
        "file", "scan", "virus", "malware", "threat", "security", "upload",
        "virustotal", "analysis", "safe", "infected", "check file", "examine",
        "secure", "verify", "detect", "dangerous", "risk file"
    }
    
    AI_RISK_KEYWORDS = {
        "ai risk", "risk database", "ai risks", "harm", "hazard", "danger ai",
        "risks related", "ai concern", "ai impact", "artificial intelligence risk",
        "misinformation", "bias", "privacy", "safety", "fairness", "robustness"
    }
    
    WEB_SEARCH_KEYWORDS = {
        "news", "search web", "internet", "current", "latest", "cybersecurity news",
        "security news", "breaches", "vulnerability", "exploit", "attack", "threat intelligence",
        "ransomware", "phishing", "zero-day", "vulnerability report", "recent news",
        "what are the latest", "what's happening", "breaking news"
    }
    
    @classmethod
    def detect_intent(cls, user_input: str) -> str:
        """
        Detect user intent from input text.
        
        Returns:
            str: One of 'file_scan', 'ai_risk_search', 'web_search', or 'general_chat'
        """
        input_lower = user_input.lower().strip()
        
        # Count keyword matches for each service
        file_score = sum(1 for kw in cls.FILE_SCAN_KEYWORDS if kw in input_lower)
        risk_score = sum(1 for kw in cls.AI_RISK_KEYWORDS if kw in input_lower)
        web_score = sum(1 for kw in cls.WEB_SEARCH_KEYWORDS if kw in input_lower)
        
        # Return highest scoring intent (or general_chat if no clear intent)
        max_score = max(file_score, risk_score, web_score)
        
        if max_score == 0:
            return "general_chat"
        
        if file_score == max_score and file_score > 0:
            return "file_scan"
        elif risk_score == max_score and risk_score > 0:
            return "ai_risk_search"
        elif web_score == max_score and web_score > 0:
            return "web_search"
        else:
            return "general_chat"
    
    @classmethod
    def get_redirect_message(cls, intent: str, user_input: str = "") -> str:
        """Generate helpful redirect message"""
        messages = {
            "file_scan": "👉 Looks like you want to scan a file! Go to the **🔒 File Scanner** tab to upload and analyze files for threats.",
            "ai_risk_search": "👉 Looks like you're asking about AI risks! Go to the **🧠 AI Risk Search** tab or use the **Quick Search** above to find information from our AI Risk Database.",
            "web_search": "👉 Looks like you want the latest news! Go to the **🌐 Security News Search** tab to search the web for current cybersecurity news and breaches.",
            "general_chat": "I'm here to help! Ask me anything or try one of these features:\n- **💬 Chat**: General conversation\n- **🔒 File Scanner**: Check file security\n- **🧠 AI Risk Search**: Query our AI Risk Database\n- **🌐 Security News Search**: Get latest security news"
        }
        return messages.get(intent, messages["general_chat"])


intent_detector = IntentDetector()
print("✅ Intent detection system initialized")
print(f"   📁 File scan keywords: {len(IntentDetector.FILE_SCAN_KEYWORDS)}")
print(f"   🧠 AI Risk keywords: {len(IntentDetector.AI_RISK_KEYWORDS)}")
print(f"   🌐 Web search keywords: {len(IntentDetector.WEB_SEARCH_KEYWORDS)}")


# ============ SERVICE IMPORTS ============
sys.path.insert(0, os.path.dirname(__file__))

WEB_SEARCH_MODE = "unavailable"
_security_search_fn = None
_basic_web_search_fn = None

try:
    from service_1 import scan_file_with_details, VIRUSTOTAL_API_KEY
    from service_2 import hybrid_search, format_search_results_markdown

    scan_file_with_virustotal = scan_file_with_details
    print(f"✅ VirusTotal API Key loaded: {VIRUSTOTAL_API_KEY[:3]}...")
    print("✅ VirusTotal scanner imported from service_1.py")
    print("✅ AI Risk hybrid search service imported from service_2.py")
except ImportError as e:
    print(f"⚠️ Warning: Could not import services: {e}")
    VIRUSTOTAL_API_KEY = "demo_key"

try:
    from service_3 import search_security_news_with_llm as _security_search_fn
    WEB_SEARCH_MODE = "llm"
    print("✅ Web search + LLM service imported from service_3.py")
except Exception:
    try:
        from service_3 import web_search as _basic_web_search_fn
        WEB_SEARCH_MODE = "basic"
        print("✅ Basic web search service imported from service_3.py")
    except Exception as e:
        print(f"⚠️ Warning: Could not import service_3 web search: {e}")


# ============ CHAT FUNCTIONS ============
def interpret_scan_results_with_llm(scan_json: dict) -> str:
    """
    Use the LLM to interpret and explain VirusTotal scan results to the user.
    
    Args:
        scan_json: The JSON result from VirusTotal scan
        
    Returns:
        LLM's natural language interpretation of the scan
    """
    try:
        if not isinstance(scan_json, dict):
            return f"⚠️ Invalid scan data type: {type(scan_json).__name__}"
        
        file_info = scan_json.get('file_info', {})
        if not isinstance(file_info, dict):
            file_info = {}
        
        scan_info = scan_json.get('scan_info', {})
        if not isinstance(scan_info, dict):
            scan_info = {}
        
        try:
            file_name = str(file_info.get('name', 'Unknown')).strip() or 'Unknown'
            file_size = str(file_info.get('size_bytes', 'Unknown')).strip() or 'Unknown'
            file_hash = str(file_info.get('sha256_hash', 'Unknown')).strip() or 'Unknown'
            scan_status = str(scan_info.get('status', 'Unknown')).strip() or 'Unknown'
            scan_id = str(scan_info.get('scan_id', 'Unknown')).strip() or 'Unknown'
        except (ValueError, TypeError):
            return f"⚠️ Error processing scan data: Could not parse file information"
        
        try:
            scan_summary = f"""
**File Scan Summary:**
- File: {file_name}
- Size: {file_size} bytes
- SHA256: {file_hash}
- Scan Status: {scan_status}
- Scan ID: {scan_id}

Please provide a clear, concise interpretation of this file scan result for a non-technical user. 
Explain what the status means, what they should expect, and any recommended actions.
"""
        except (TypeError, ValueError):
            return "⚠️ Error formatting scan summary for analysis"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert explaining file scan results in simple, non-technical language. Be clear, concise, and actionable."
                    },
                    {
                        "role": "user",
                        "content": scan_summary
                    }
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            interpretation = str(response.choices[0].message.content).strip()
        except Exception as llm_error:
            return f"⚠️ Could not get LLM interpretation: {type(llm_error).__name__}"
        
        try:
            is_valid, result = guardrails.validate_output(interpretation)
            return result
        except GuardrailViolation:
            return interpretation
            
    except Exception as e:
        try:
            error_msg = str(e)
        except:
            error_msg = type(e).__name__
        return f"⚠️ Could not interpret results: {error_msg}"


# Config storage
config_choices = {
    "personality": "default",
    "tone": "friendly", 
    "style": "concise",
    "temperature": 0.7
}


def simple_chat(message, history):
    """Smart chat function with intent detection that executes services directly"""
    try:
        try:
            guardrails.validate_input(message, user_id="gradio_user")
        except GuardrailViolation as gv:
            return f"⛔ Request blocked: {gv.message}"
        
        # Detect user intent
        intent = intent_detector.detect_intent(message)
        
        # ===== SERVICE 2: AI Risk Search (Can execute directly from chat) =====
        if intent == "ai_risk_search":
            try:
                results = hybrid_search(message, top_k=5)
                markdown = format_search_results_markdown(results, message)
                return f"## 🧠 AI Risk Search Results\n\n{markdown}"
            except Exception as e:
                return f"⚠️ Search failed: {str(e)}"
        
        # ===== SERVICE 3: Web Search (Can execute directly from chat) =====
        elif intent == "web_search":
            try:
                if WEB_SEARCH_MODE == "llm" and _security_search_fn:
                    result = _security_search_fn(message, max_results=5)
                    return result
                elif WEB_SEARCH_MODE == "basic" and _basic_web_search_fn:
                    result = _basic_web_search_fn(message, max_results=5)
                    return f"## 🌐 Web Search Results\n\n{result}"
                else:
                    return "⚠️ Web search service unavailable. Try the 🌐 Security News Search tab instead."
            except Exception as e:
                return f"⚠️ Web search failed: {str(e)}"
        
        # ===== SERVICE 1: File Scanner (Requires file upload - must use tab) =====
        elif intent == "file_scan":
            return (
                "🔒 **File Scanning via VirusTotal**\n\n"
                "To scan a file, please use the **🔒 File Scanner** tab where you can:\n"
                "1. Upload your file\n"
                "2. Click 'Scan & Analyze'\n"
                "3. See AI interpretation + technical details\n\n"
                "_File scanning requires file upload which is better handled in the dedicated tab._"
            )
        
        # ===== GENERAL CHAT =====
        else:
            temp = config_choices.get("temperature", 0.7)
            if not (0.0 <= temp <= 2.0):
                return "⛔ Temperature out of valid range (0.0-2.0)"
            
            personalities = PersonalityLibrary.get_all_personalities()
            base_config = personalities.get(config_choices["personality"], PersonalityLibrary.DEFAULT)
            
            tone_str = config_choices["tone"]
            style_str = config_choices["style"]
            
            system_prompt = f"""You are a helpful AI assistant with a {tone_str} tone.
Your conversational style is {style_str}.

Guidelines:
- Do not answer any requests on the following topics: Cats, Dogs, Horoscopes, Zodiac Signs, or Taylor Swift.
- Be helpful, accurate, and respectful."""
            
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": message})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temp,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            try:
                is_valid, result = guardrails.validate_output(assistant_response)
                return result if is_valid else assistant_response
            except GuardrailViolation as gv:
                return "⛔ Response blocked by content policy"
    
    except GuardrailViolation as gv:
        return f"⛔ [{gv.severity}] {gv.message}"
    except Exception as e:
        return f"Error: {str(e)}"


def run_ai_risk_search(query, top_k=5, lexical_k=50, alpha=0.35):
    """Run hybrid lexical + semantic search against the AI Risk Database."""
    if not query or not query.strip():
        return "❌ Please enter a question.", []
    results = hybrid_search(query, top_k=top_k, lexical_k=lexical_k, alpha=alpha)
    markdown = format_search_results_markdown(results, query)
    return markdown, results


def landing_ai_risk_search(query, top_k=5):
    """Landing-page helper that also populates the AI Risk Search tab."""
    if not query or not query.strip():
        return "❌ Please enter a question.", "", "❌ Please enter a question.", []
    results = hybrid_search(query, top_k=top_k)
    markdown = format_search_results_markdown(results, query)
    landing_message = f"✅ Results loaded in **AI Risk Search** tab.\n\n{markdown}"
    return landing_message, query, markdown, results


def run_security_news_search(prompt, top_k=5):
    """Run security-focused web search using service_3."""
    if not prompt or not prompt.strip():
        return "❌ Please enter a search prompt."

    try:
        guardrails.validate_input(prompt, user_id="gradio_web_search_user")
    except GuardrailViolation as gv:
        return f"⛔ Request blocked: {gv.message}"

    try:
        if WEB_SEARCH_MODE == "llm" and _security_search_fn:
            return _security_search_fn(prompt, max_results=int(top_k))

        if WEB_SEARCH_MODE == "basic" and _basic_web_search_fn:
            return f"## 🌐 Web Search Results\n\n{_basic_web_search_fn(prompt)}"

        return "⚠️ Web search service is unavailable. Check service_3.py configuration."
    except Exception as e:
        return f"⚠️ Web search failed: {type(e).__name__}: {e}"


print("✅ Chat functions defined")

# ============ BUILD GRADIO INTERFACE ============
def build_interface():
    """Build the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Walter's AI Chatbot, Security Scanner and Risk Modules")
        gr.Markdown("Customize your chat experience or scan files for security threats")

        gr.Markdown("### 🔎 Quick Search the AI Risk Database")
        with gr.Row():
            landing_query = gr.Textbox(
                label="Ask about AI risks",
                placeholder="e.g., What are risks related to misinformation?"
            )
            landing_top_k = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Top results"
            )
            landing_search_btn = gr.Button("Search", variant="secondary")
        landing_output = gr.Markdown("")
        
        with gr.Tabs():
            # ===== TAB 1: Chat Interface =====
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    personality = gr.Dropdown(
                        choices=list(PersonalityLibrary.get_all_personalities().keys()),
                        value="default",
                        label="🎭 Personality"
                    )
                    tone = gr.Dropdown(
                        choices=[t.value for t in ToneType],
                        value="friendly",
                        label="🎵 Tone"
                    )
                    style = gr.Dropdown(
                        choices=[s.value for s in StyleType],
                        value="concise",
                        label="📝 Style"
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="🔥 Temperature"
                    )
                
                personality.change(lambda x: config_choices.update({"personality": x}), inputs=personality)
                tone.change(lambda x: config_choices.update({"tone": x}), inputs=tone)
                style.change(lambda x: config_choices.update({"style": x}), inputs=style)
                temperature.change(lambda x: config_choices.update({"temperature": x}), inputs=temperature)
                
                chatbot = gr.ChatInterface(
                    fn=simple_chat,
                    examples=[
                        "How can i scan my files with viruses?",
                        "What are the top categories of AI risks as per MIT??",
                        "Tell me the latest cybersecurity news today?",
                        "What is the capital of France?"
                    ],
                    type="messages"
                )
            
            # ===== TAB 2: File Scanner =====
            with gr.Tab("🔒 File Scanner"):
                gr.Markdown("### VirusTotal File Security Scan with AI Interpretation")
                gr.Markdown("Upload a file to scan it for security threats. Results will be analyzed and explained by AI.")
                
                with gr.Row():
                    file_input = gr.File(
                        label="📁 Select File to Scan",
                        type="filepath"
                    )
                    scan_button = gr.Button("🔍 Scan & Analyze", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        ai_interpretation = gr.Markdown("Upload a file and click 'Scan & Analyze' to begin analysis")
                    with gr.Column():
                        scan_output = gr.Markdown("Scan details will appear here")
                
                with gr.Row():
                    json_output = gr.JSON(label="📊 Raw Scan Data (JSON)", visible=False)
                
                def scan_and_interpret(file_path):
                    """Scan file and get LLM interpretation of results"""
                    if not file_path:
                        return "❌ Please select a file", "No file selected", {}
                    
                    markdown_result, json_result = scan_file_with_virustotal(file_path)
                    
                    if json_result and json_result.get("status") == "success":
                        llm_interpretation = interpret_scan_results_with_llm(json_result)
                        ai_output = f"## 🤖 AI Analysis\n\n{llm_interpretation}\n\n---\n\n## 📋 Technical Details\n\n{markdown_result}"
                    else:
                        ai_output = markdown_result
                    
                    return ai_output, markdown_result, json_result
                
                scan_button.click(
                    fn=scan_and_interpret,
                    inputs=file_input,
                    outputs=[ai_interpretation, scan_output, json_output]
                )

            # ===== TAB 3: AI Risk Hybrid Search =====
            with gr.Tab("🧠 AI Risk Search"):
                gr.Markdown("### AI Risk Database Hybrid Search")
                gr.Markdown("Lexical pre-filtering + semantic re-ranking over MIT Risk Dataset")

                risk_query = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g., What risks involve privacy leakage?"
                )
                risk_top_k = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top results (adjust for more/less results)"
                )
                risk_search_btn = gr.Button("🔎 Search", variant="primary")

                risk_results_md = gr.Markdown("Enter a question and click Search to see results.")
                #this will also be populated by the landing page quick search for convenience
                risk_results_json = gr.JSON(label="📄 Raw Results (JSON)", visible=False)

                risk_search_btn.click(
                    fn=run_ai_risk_search,
                    inputs=[risk_query, risk_top_k],
                    outputs=[risk_results_md, risk_results_json]
                )

            # ===== TAB 4: Security Web Search =====
            with gr.Tab("🌐 Security News Search"):
                gr.Markdown("### LLM-Assisted Security News Web Search")
                gr.Markdown("Search the web for current cybersecurity/security news and get an AI summary.")

                web_query = gr.Textbox(
                    label="Search prompt",
                    placeholder="e.g., latest ransomware attacks this week"
                )
                web_top_k = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top web results to use"
                )
                web_search_btn = gr.Button("🌐 Search Web", variant="primary")
                web_results_md = gr.Markdown("Enter a prompt and click Search Web.")

                web_search_btn.click(
                    fn=run_security_news_search,
                    inputs=[web_query, web_top_k],
                    outputs=[web_results_md]
                )

        landing_search_btn.click(
            fn=landing_ai_risk_search,
            inputs=[landing_query, landing_top_k],
            outputs=[landing_output, risk_query, risk_results_md, risk_results_json]
        )
    
    return demo


print("✅ Gradio interface built")

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LAUNCHING CHATBOT APPLICATION".center(70))
    print("="*70)
    print()
    print("📱 Open your browser and navigate to: http://localhost:7860")
    print()
    print("✨ Features:")
    print("   • Full conversation history")
    print("   • Real-time responses from GPT-4")
    print("   • Amazing, responsive interface")
    print("   • One-click chat management")
    print("   • File security scanning with AI analysis")
    print("   • AI Risk Database hybrid search")
    print("   • Web security news search (service_3)")
    print()
    print("⏹️  Press Ctrl+C in terminal to stop")
    print()
    print("="*70 + "\n")
    
    # Build and launch
    demo = build_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=None,
        show_error=True,
        quiet=False
    )
