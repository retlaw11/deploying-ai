## This file will be an AI service that performs a websearch to retrieve the latest security news

import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import quote_plus

# Load environment variables BEFORE accessing them
load_dotenv('./.secrets')
load_dotenv('../../../.env')  # Fallback to root .env if needed

API_GATEWAY_KEY = os.getenv('API_GATEWAY_KEY')
if not API_GATEWAY_KEY:
    print("⚠️ Warning: API_GATEWAY_KEY not found. Running in demo mode.")
    API_GATEWAY_KEY = "demo_key"

# Initialize OpenAI client for LLM-powered synthesis
client = OpenAI(
    default_headers={"x-api-key": API_GATEWAY_KEY},
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1'
)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Fetch real-time news from Google News RSS feed focused on security and AI topics.
    
    Args:
        query: Search query string (e.g., "cybersecurity AI threats")
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted string with news articles (title, link, published date, snippet)
    """
    try:
        # Enhance query with security/AI focus if not already present
        enhanced_query = query.lower()
        if "security" not in enhanced_query and "cyber" not in enhanced_query and "ai" not in enhanced_query:
            enhanced_query = f"{query} security OR AI"
        
        # Google News RSS URL format
        encoded_query = quote_plus(enhanced_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        # Fetch RSS feed with timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Extract articles
        articles = []
        for item in root.findall('.//item')[:max_results]:
            title = item.find('title')
            link = item.find('link')
            pub_date = item.find('pubDate')
            description = item.find('description')
            
            article = {
                'title': title.text if title is not None else 'No title',
                'link': link.text if link is not None else '',
                'published': pub_date.text if pub_date is not None else 'Unknown date',
                'snippet': description.text if description is not None else 'No description'
            }
            articles.append(article)
        
        # Format results
        if not articles:
            return f"No recent news found for query: '{query}'"
        
        result = f"Found {len(articles)} recent articles:\n\n"
        for i, article in enumerate(articles, 1):
            result += f"{i}. **{article['title']}**\n"
            result += f"   📅 {article['published']}\n"
            result += f"   🔗 {article['link']}\n"
            if article['snippet'] and article['snippet'] != 'No description':
                # Clean HTML tags from snippet
                snippet = article['snippet'].replace('<b>', '').replace('</b>', '')
                snippet = snippet.replace('<br>', ' ').replace('&nbsp;', ' ')
                result += f"   📝 {snippet[:200]}...\n"
            result += "\n"
        
        return result
        
    except requests.RequestException as e:
        return f"⚠️ Network error fetching news: {str(e)}"
    except ET.ParseError as e:
        return f"⚠️ Error parsing RSS feed: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


def search_security_news_with_llm(prompt: str, max_results: int = 5) -> str:
    """
    Fetch security/AI news from Google News RSS and use LLM to synthesize and analyze results.
    
    Args:
        prompt: User's search prompt or question
        max_results: Maximum number of articles to fetch
    
    Returns:
        LLM-generated analysis and summary of the news articles
    """
    try:
        # Fetch raw news data
        raw_results = web_search(prompt, max_results)
        
        # If web search failed, return the error
        if raw_results.startswith("⚠️"):
            return raw_results
        
        # Use LLM to synthesize and analyze the results
        system_message = """You are a security and AI news analyst. Your task is to:
1. Analyze the provided news articles
2. Summarize key trends and important developments
3. Highlight any critical security threats or AI breakthroughs
4. Provide actionable insights for security professionals

Format your response in markdown with clear sections."""

        user_message = f"""User Query: {prompt}

News Articles Retrieved:
{raw_results}

Please analyze these articles and provide:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Notable Trends
- Security Implications (if applicable)
- Recommendations or Next Steps"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        llm_analysis = response.choices[0].message.content
        
        # Combine original sources with LLM analysis
        final_output = f"## 🔍 Security & AI News Analysis\n\n"
        final_output += f"**Query:** {prompt}\n\n"
        final_output += "---\n\n"
        final_output += llm_analysis
        final_output += "\n\n---\n\n"
        final_output += "### 📰 Source Articles\n\n"
        final_output += raw_results
        
        return final_output
        
    except Exception as e:
        return f"⚠️ Error during LLM analysis: {str(e)}\n\nRaw results:\n{raw_results if 'raw_results' in locals() else 'No results fetched'}"



if __name__ == "__main__":
    print("✅ Service 3 (Web Search Service) initialized")
    print(f"   API Gateway Key: {API_GATEWAY_KEY[:5]}..." if API_GATEWAY_KEY != "demo_key" else "   Running in demo mode")
    print("\n🧪 Testing web search functionality...\n")
    
    # Test 1: Basic web search
    print("=" * 60)
    print("Test 1: Basic Web Search")
    print("=" * 60)
    test_query = "AI cybersecurity threats 2026"
    print(f"Query: {test_query}\n")
    basic_results = web_search(test_query, max_results=3)
    print(basic_results)
    
    # Test 2: LLM-enhanced search
    print("\n" + "=" * 60)
    print("Test 2: LLM-Enhanced Search")
    print("=" * 60)
    test_prompt = "What are the latest AI security vulnerabilities?"
    print(f"Prompt: {test_prompt}\n")
    llm_results = search_security_news_with_llm(test_prompt, max_results=3)
    print(llm_results)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)