## Assignment 2 Details

This is assignment_chat UfT deploying ai assignment #2. 
Presenter/Owner: Walter Pareja
Contact: retlaw1@gmail.com
@Github Handle: retlaw11 
----------


Functional user flows
- User can submit prompt via text form to query about stock market data. Some questions that can be asked include
Q1. What is the current stock price of Microsoft, Amazon, etc. 
Q2. How much volume was traded

Answers are transformed using
* Pydantic models, structured JSON output
* I have defined the Pydantic schemal 


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

