"""Small examples that show how to use the Google GenAI LLM wrapper with LangChain.

Run:
    python3 sample_langchain.py

Requirements: installed `langchain` and `google-genai` and set GEMINI_API_KEY or ADC.
"""
# Load .env FIRST (before any other imports that might use env vars)
from dotenv import load_dotenv

load_dotenv()

# Try to import LangChain features; if they're not available, provide graceful fallbacks
try:
    from langchain.chains import LLMChain  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain.agents import initialize_agent  # type: ignore
    from langchain.tools import Tool  # type: ignore
    HAS_LANGCHAIN = True
except Exception:
    LLMChain = None
    PromptTemplate = None
    initialize_agent = None
    Tool = None
    HAS_LANGCHAIN = False

from langsmith import traceable
from langchain_google_genai import GoogleGenAILLM


def calculator(expr: str) -> str:
    """Tiny safe evaluator for simple arithmetic expressions (no statements).
    It supports +, -, *, /, **, (), and integers/floats.

    This function is defensive: it strips whitespace and raises a readable
    error if parsing fails so callers can fallback to other handlers.
    """
    import ast
    import operator as op

    allowed_operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def _eval(node):
        # ast.Num is deprecated in newer Python AST versions; support ast.Constant
        from ast import Constant

        if isinstance(node, Constant):
            return node.value
        # Fallback for very old Python versions (pre-3.8)
        if hasattr(node, "n"):
            return node.n
        if isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    try:
        node = ast.parse(expr, mode="eval").body
    except Exception as e:
        raise ValueError(f"Invalid expression {expr!r}: {e}")

    return str(_eval(node))


@traceable
def chain_example():
    llm = GoogleGenAILLM(model="gemini-2.5-flash")

    if HAS_LANGCHAIN and LLMChain and PromptTemplate:
        prompt = PromptTemplate(input_variables=["question"], template="Answer concisely: {question}")
        chain = LLMChain(llm=llm, prompt=prompt)

        print("--- LLMChain example ---")
        print(chain.run("What is the capital of France?"))
    else:
        # Fallback: call the LLM directly
        print("--- LLM fallback example ---")
        print(llm("Answer concisely: What is the capital of France?"))

@traceable
def agent_example():
    llm = GoogleGenAILLM(model="gemini-2.5-flash")

    if HAS_LANGCHAIN and initialize_agent and Tool:
        tools = [
            Tool(name="calculator", func=calculator, description="Evaluate math expressions"),
        ]

        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

        print("--- Agent example (calculator via LangChain) ---")
        print(agent.run("What is 12 * (7 + 3)?"))
    else:
        # Simple fallback agent: naive routing
        print("--- Agent fallback (simple router) ---")
        q = "What is 12 * (7 + 3)?"
        # Very simple heuristic: if question contains digits or math symbols, use calculator
        import re

        if re.search(r"[0-9\+\-\*/\(\)]", q):
            expr = re.sub(r"[^0-9\+\-\*/\(\)\. ]", "", q)
            print("Calculator result:", calculator(expr))
        else:
            print("LLM result:", llm(q))


if __name__ == "__main__":
    chain_example()
    agent_example()
