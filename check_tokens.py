"""Check token fields."""
from dotenv import load_dotenv
load_dotenv()

import os
import google.genai as genai

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 2+2? Answer in one sentence."
)

usage = response.usage_metadata
print("Token Information:")
print(f"  prompt_token_count: {usage.prompt_token_count}")
print(f"  candidates_token_count: {usage.candidates_token_count}")
print(f"  total_token_count: {usage.total_token_count}")
print(f"  cached_content_token_count: {usage.cached_content_token_count}")
