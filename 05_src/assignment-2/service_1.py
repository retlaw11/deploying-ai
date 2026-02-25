"""
VirusTotal API Service
Scans files/URLs using the VirusTotal API and processes reports with LLM.
Returns easy-to-read security analysis for users.
Can be expanded with threat intelligence from incidents, events, and zero-day malware.
"""

import requests
import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .secrets file
load_dotenv('./.secrets')

# Get VirusTotal API key from environment
VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')

if not VIRUSTOTAL_API_KEY:
    raise ValueError("VIRUSTOTAL_API_KEY not found in .secrets file")

# VirusTotal API endpoints
VIRUSTOTAL_BASE_URL = "https://www.virustotal.com/api/v3"
FILE_SCAN_URL = f"{VIRUSTOTAL_BASE_URL}/files"
URL_SCAN_URL = f"{VIRUSTOTAL_BASE_URL}/urls"
ANALYSIS_URL = f"{VIRUSTOTAL_BASE_URL}/analyses"

# API headers with authentication
headers = {
    "accept": "application/json",
    "x-apikey": VIRUSTOTAL_API_KEY
}

print("✅ VirusTotal API configured successfully")
print(f"   API Key: {VIRUSTOTAL_API_KEY[:20]}..." if VIRUSTOTAL_API_KEY else "❌ API Key not loaded")
print(f"   Base URL: {VIRUSTOTAL_BASE_URL}")


# ============ UTILITY FUNCTIONS ============

def get_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of file
    
    Args:
        file_path: Path to file
        
    Returns:
        str: SHA256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ============ FILE SCANNING ============

def scan_file(file_path: str) -> dict:
    """
    Scan a file using VirusTotal API (basic version)
    
    Args:
        file_path: Path to file to scan
        
    Returns:
        dict: API response with scan results
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                FILE_SCAN_URL,
                headers=headers,
                files=files
            )
        response.raise_for_status()
        result = response.json()
        print(f"✅ File scan initiated: {result.get('data', {}).get('id', 'N/A')}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error scanning file: {e}")
        return {"error": str(e)}


def scan_file_with_details(file_path: str) -> tuple:
    """
    Scan a file using VirusTotal API with structured full results
    
    Args:
        file_path: Path to file to scan
        
    Returns:
        tuple: (markdown_summary, json_data)
    """
    if not file_path:
        return "❌ No file selected", None
    
    try:
        # Get file info
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_hash = get_file_hash(file_path)
        
        print(f"📁 Scanning file: {filename}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   SHA256: {file_hash}")
        
        # Scan the file
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                FILE_SCAN_URL,
                headers=headers,
                files=files
            )
        
        response.raise_for_status()
        result = response.json()
        
        if "data" in result:
            scan_id = result["data"]["id"]
            timestamp = datetime.now().isoformat()
            
            # Create structured result
            structured_result = {
                "status": "success",
                "timestamp": timestamp,
                "file_info": {
                    "name": filename,
                    "size_bytes": file_size,
                    "sha256_hash": file_hash
                },
                "scan_info": {
                    "scan_id": scan_id,
                    "status": "queued",
                    "virustotal_link": f"https://www.virustotal.com/gui/file/{scan_id}"
                }
            }
            
            # Create readable summary
            summary = f"""# ✅ File Scan Initiated

## 📄 File Information
- **Name**: `{filename}`
- **Size**: {file_size:,} bytes
- **SHA256**: `{file_hash}`

## 🔍 Scan Details
- **Scan ID**: `{scan_id}`
- **Status**: ⏳ Queued for analysis
- **Timestamp**: {timestamp}

## 🔗 View Results
[Open in VirusTotal](https://www.virustotal.com/gui/file/{scan_id})

---

### Full Scan Data (JSON)
```json
{json.dumps(structured_result, indent=2)}
```
            """
            
            print(f"✅ File scan initiated successfully: {scan_id}")
            return summary, structured_result
        else:
            error_result = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Unexpected API response",
                "response": result
            }
            return f"❌ Unexpected response: {result}", error_result
    
    except FileNotFoundError:
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"File not found: {file_path}"
        }
        print(f"❌ File not found: {file_path}")
        return f"❌ File not found: {file_path}", error_result
    except requests.exceptions.RequestException as e:
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"API Error: {str(e)}"
        }
        print(f"❌ API Error: {str(e)}")
        return f"❌ API Error: {str(e)}", error_result
    except Exception as e:
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Error scanning file: {str(e)}"
        }
        print(f"❌ Error scanning file: {str(e)}")
        return f"❌ Error scanning file: {str(e)}", error_result


# ============ URL SCANNING ============

def scan_url(url: str) -> dict:
    """
    Scan a URL using VirusTotal API
    
    Args:
        url: URL to scan
        
    Returns:
        dict: API response with scan results
    """
    payload = {"url": url}
    
    try:
        print(f"🌐 Scanning URL: {url}")
        response = requests.post(
            URL_SCAN_URL,
            headers=headers,
            data=payload
        )
        response.raise_for_status()
        result = response.json()
        scan_id = result.get('data', {}).get('id', 'N/A')
        print(f"✅ URL scan initiated: {scan_id}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error scanning URL: {e}")
        return {"error": str(e)}


# ============ RESULTS RETRIEVAL ============

def get_scan_results(scan_id: str) -> dict:
    """
    Get results of a previously submitted scan
    
    Args:
        scan_id: ID of the scan (from scan_url or scan_file response)
        
    Returns:
        dict: Scan results with threat analysis
    """
    try:
        print(f"🔍 Fetching results for scan: {scan_id}")
        response = requests.get(
            f"{ANALYSIS_URL}/{scan_id}",
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Results retrieved successfully")
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting scan results: {e}")
        return {"error": str(e)}


# ============ ANALYSIS HELPERS ============

def format_scan_results(scan_result: dict) -> str:
    """
    Format scan results in a human-readable way
    
    Args:
        scan_result: Raw scan result from get_scan_results
        
    Returns:
        str: Formatted summary
    """
    if "error" in scan_result:
        return f"❌ Error: {scan_result['error']}"
    
    try:
        data = scan_result.get('data', {})
        attributes = data.get('attributes', {})
        stats = attributes.get('stats', {})
        
        timestamp = attributes.get('date', 'Unknown')
        
        summary = f"""# 📊 VirusTotal Scan Results

## Detection Stats
- **Malicious**: {stats.get('malicious', 0)} engines
- **Suspicious**: {stats.get('suspicious', 0)} engines
- **Undetected**: {stats.get('undetected', 0)} engines
- **Harmless**: {stats.get('harmless', 0)} engines

## Analysis
- **Scan Date**: {timestamp}
- **Last Analysis Stats**: {json.dumps(stats, indent=2)}
        """
        return summary
    except Exception as e:
        return f"❌ Error parsing results: {str(e)}"


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Example 1: Test file scanning with details
    print("\n" + "="*70)
    print("TESTING FILE SCAN WITH STRUCTURED RESULTS".center(70))
    print("="*70)
    
    # Create a test file
    test_file = "/tmp/test_virustotal.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test file for VirusTotal scanning.")
    
    print(f"\n✅ Created test file: {test_file}\n")
    
    # Scan with detailed results
    markdown_result, json_result = scan_file_with_details(test_file)
    print("\n--- Markdown Summary ---")
    print(markdown_result)
    print("\n--- JSON Result ---")
    print(json.dumps(json_result, indent=2) if json_result else "None")
    
    # Example 2: Test URL scanning
    print("\n" + "="*70)
    print("TESTING URL SCAN".center(70))
    print("="*70)
    test_url = "https://www.google.com"
    url_result = scan_url(test_url)
    if "data" in url_result:
        scan_id = url_result["data"]["id"]
        print(f"\nScan ID: {scan_id}")
        print("To get results later, use:")
        print(f"  result = get_scan_results('{scan_id}')")
    
    # Clean up
    os.remove(test_file)
    
    print("\n" + "="*70)
    print("✅ VirusTotal Service Ready!".center(70))
    print("="*70)
    print("\nAvailable functions:")
    print("  • scan_file(file_path) - Scan a file (basic)")
    print("  • scan_file_with_details(file_path) - Scan a file (structured)")
    print("  • scan_url(url) - Scan a URL")
    print("  • get_scan_results(scan_id) - Get scan results")
    print("  • format_scan_results(scan_result) - Format results nicely")