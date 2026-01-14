"""Debug script to check token usage from Gemini API."""
from dotenv import load_dotenv
load_dotenv()

import os
import google.genai as genai

# Initialize client
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Make a simple API call and capture token usage
print("Making API call to Gemini...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 2+2? Answer in one sentence."
)

print(f"\nResponse Text: {response.text}")
print(f"\nResponse attributes: {dir(response)}")

# Check for usage_metadata
if hasattr(response, 'usage_metadata'):
    print(f"\n✓ usage_metadata found!")
    usage = response.usage_metadata
    print(f"  Type: {type(usage)}")
    print(f"  Attributes: {dir(usage)}")
    
    if hasattr(usage, 'input_tokens'):
        print(f"  Input tokens: {usage.input_tokens}")
    if hasattr(usage, 'output_tokens'):
        print(f"  Output tokens: {usage.output_tokens}")
    if hasattr(usage, 'total_tokens'):
        print(f"  Total tokens: {usage.total_tokens}")
else:
    print("\n✗ No usage_metadata found in response")

# Check LangSmith environment
print("\n--- LangSmith Configuration ---")
print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"LANGSMITH_API_KEY exists: {bool(os.getenv('LANGSMITH_API_KEY'))}")
