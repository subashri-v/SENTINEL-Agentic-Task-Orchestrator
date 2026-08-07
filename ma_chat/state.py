from typing import List, Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    next_agent: str
    answer: str
    citations: List[str]
    retry_count: int
    system_critique: str


class EvalSchema(TypedDict):
    passed: bool
    reasoning: str
