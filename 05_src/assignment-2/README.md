## Assignment 2 Details

This is assignment_chat UfT deploying ai assignment #2. 
Presenter/Owner: Walter Pareja
Contact: retlaw1@gmail.com
@Github Handle: retlaw11 
----------

assignment-2/
├── service_1.py          ← Core business logic (production-ready)
├── app.ipynb             ← UI layer (imports from service)
└── .secrets              ← Configuration

🎯 LLM-Powered Scan Results
Your file scanner now has three-layer analysis:

1. VirusTotal Scanning ✅
Uploads file to VirusTotal API
Gets scan ID and file metadata (SHA256, size, etc.)
2. LLM Interpretation ✅
New interpret_scan_results_with_llm() function:

Passes scan data to your GPT model
LLM analyzes and explains results in plain English
Applies enterprise guardrails to interpretation output
Returns user-friendly summary
3. Gradio UI Integration ✅
File uploaded → Scan initiated → Results interpreted
3-panel output:
🤖 AI Analysis (LLM interpretation in natural language)
📋 Technical Details (raw scan data from VirusTotal)
📊 Raw JSON (structured data for developers).




Service 2: Semantic Query 
Using MIT AI Risk Initiative Risk Repository (downloaded and stored under this folder)
I will chunckit, create embeddings, and query against it for specific risk findings.



Service 3: 
Search the web for top 3 cybersecurity ai news articles. 
API - regulatory conditions of API 
Function calling - call the API (google maps, goog earth) to tell me where it is th
MCP Server - email me the results 



### User Interface
* The system is using a chat-based interface provided by Gradio [check]
* Chat Client will be a comedian personality defined by attributes {tone}, {role}, {conversational style} [check]
* MEMORY throughout the conversation.




##Guardrails

Users are prevented from
* Accessing or revealing the system prompt
--- This is done by:
1. 


Modying the system prompt directly
--- This is done by:
1. 


The model does not respond to questions on the following topics
1. Cats or Dogs
2. Horospoce or Zodiac Signs
3. Taylor Swift

The model does this by defined system prompts
-- 


Metrics and validation
testing 
-- accuracy to mdoel system prompts
-- accuracy to responses from the users
-- assessing system prompt testing: 
-- revealing system prompts

