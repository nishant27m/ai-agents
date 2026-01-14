"""Test script with explicit LangSmith tracing."""
from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable
from langchain_google_genai import GoogleGenAILLM

@traceable(name="Test LLM Call", run_type="chain")
def test_llm():
    llm = GoogleGenAILLM(model="gemini-2.5-flash")
    result = llm("What is the capital of France?")
    print(f"LLM Output: {result}")
    return result

if __name__ == "__main__":
    print("Running LLM with LangSmith tracing...")
    test_llm()
    print("\nDone! Check LangSmith UI at https://smith.langchain.com/projects/ai-agent")
