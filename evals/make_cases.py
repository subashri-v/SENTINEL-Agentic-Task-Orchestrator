"""Regenerates evals/cases.jsonl. Edit the lists below, then run: python evals/make_cases.py"""
import json
from pathlib import Path

cases = []


def add(category, question, **extra):
    cases.append({"id": f"{category.lower()}-{sum(c['category'] == category for c in cases) + 1:02d}",
                  "category": category, "question": question, **extra})


# MATH: graded programmatically (all substrings must appear in the answer, whitespace-insensitive)
for q, exp in [
    ("What is 2**10 * 5.5?", ["5632"]),
    ("Solve 3*x**2 + 2*x - 5 = 0 for x.", ["-5/3", "1"]),
    ("Find the derivative of x**3 + 2*x with respect to x.", ["3*x**2"]),
    ("What is the indefinite integral of x**2 with respect to x?", ["x**3/3"]),
    ("Find the roots of x**2 - 9.", ["-3", "3"]),
    ("Differentiate x*sin(x) with respect to x.", ["x*cos(x)", "sin(x)"]),
    ("Factor x**2 - 5*x + 6.", ["(x-3)*(x-2)"]),
    ("What is 17 multiplied by 23?", ["391"]),
    ("What is the square root of 1764?", ["42"]),
    ("Expand (x + 1)**3.", ["x**3+3*x**2+3*x+1"]),
    ("Solve 2*x + 6 = 0.", ["-3"]),
    ("Differentiate exp(2*x) with respect to x.", ["2*exp(2*x)"]),
]:
    add("MATH", q, expect_all=exp)

# TAVILY: route-only (answers change over time)
for q in [
    "What are the latest headlines in AI news this week?",
    "Who won the most recent Formula 1 Grand Prix?",
    "What is the current weather forecast in Chennai?",
    "Search the web for the newest release of the Python programming language.",
    "What happened in the stock market today?",
    "Look up recent news about the SpaceX Starship launch.",
    "What is the latest iPhone model that Apple has announced?",
    "Find recent news on renewable energy policy in the EU.",
    "What are people saying online about the newest OpenAI announcement?",
    "Who is leading the latest cricket World Cup tournament right now?",
]:
    add("TAVILY", q)

# GITHUB: route-only (live repo data). Two cases test the follow-up rewriter.
for q in [
    "What are the open issues in the pallets/flask repository?",
    "Show me the description of the psf/requests repo.",
    "List the open pull requests for langchain-ai/langgraph.",
    "How many repositories do I have on my GitHub account?",
    "What is the latest issue opened in tiangolo/fastapi?",
    "Give me details about the microsoft/vscode repo.",
    "Are there any open bug issues in django/django?",
]:
    add("GITHUB", q)
add("GITHUB", "And what are its open issues?",
    history=[["user", "Tell me about the pallets/flask repository."],
             ["assistant", "pallets/flask is a lightweight Python web framework."]])

# RAG: route-only. ADAPT these to whatever you have ingested, and label them by hand.
for q in [
    "According to my uploaded documents, what are the main conclusions?",
    "What does the ingested document say about eligibility requirements?",
    "Based on the files I uploaded, what deadlines are mentioned?",
    "Search my documents for the section on fees and tuition.",
    "What does my uploaded PDF say about the application process?",
    "Find the part of my internal files that describes the project scope.",
    "Which universities are mentioned in the documents I ingested?",
    "Summarize the key requirements listed in my uploaded resume document.",
    "What is stated in the uploaded document about contact information?",
    "Look in my documents for anything on scholarship criteria.",
]:
    add("RAG", q)

# SUMMARY_AGENT: graded by LLM against `reference`; history is seeded into the thread first.
H = [
    ([["user", "What is a Python decorator?"],
      ["assistant", "A decorator is a function that wraps another function to extend its behavior, applied with the @ syntax."]],
     "Summarize our conversation so far.",
     "The conversation was about Python decorators: a decorator wraps a function to extend its behavior, using @ syntax."),
    ([["user", "What is 12 times 12?"], ["assistant", "12 times 12 is 144."]],
     "What was the last question I asked you, and what was your answer?",
     "The user asked what 12 times 12 is, and the answer was 144."),
    ([["user", "How often should I water tomato plants?"],
      ["assistant", "Water tomato plants deeply about two to three times a week, more often in hot weather."]],
     "Recap what we have discussed in one sentence.",
     "It discussed watering tomato plants: deeply two to three times a week, more in hot weather."),
    ([["user", "Who wrote Pride and Prejudice?"], ["assistant", "Jane Austen wrote Pride and Prejudice."],
      ["user", "When was it published?"], ["assistant", "It was published in 1813."]],
     "Give me a summary of this chat session.",
     "The chat covered Pride and Prejudice: written by Jane Austen and published in 1813."),
    ([["user", "What is the capital of France?"], ["assistant", "The capital of France is Paris."]],
     "What did I ask you earlier in this chat?",
     "The user asked what the capital of France is."),
    ([["user", "Explain what an API is."],
      ["assistant", "An API is an interface that lets software programs communicate with each other."],
      ["user", "Give an example."],
      ["assistant", "A weather app calling a weather service's API to fetch the forecast."]],
     "Summarize the thread so far in bullet points.",
     "It explained that an API lets programs communicate, with the example of a weather app fetching a forecast."),
    ([["user", "Convert 5 miles to kilometers."], ["assistant", "5 miles is about 8.05 kilometers."]],
     "Remind me what conversion we just did.",
     "It converted 5 miles to about 8.05 kilometers."),
    ([["user", "What does HTTP stand for?"], ["assistant", "HTTP stands for HyperText Transfer Protocol."]],
     "Can you recap our chat so far?",
     "The chat covered that HTTP stands for HyperText Transfer Protocol."),
]
for hist, q, ref in H:
    add("SUMMARY_AGENT", q, history=hist, reference=ref)

out = Path(__file__).parent / "cases.jsonl"
out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
print(f"wrote {len(cases)} cases to {out}")
