import json
import asyncio

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from . import config
from .database import retrieve
from .state import AgentState, EvalSchema

# =====================================================
# THE NODES
# =====================================================

async def node_math_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running MCP Math Engine Agent (NVIDIA)...")
    question = state["question"]

    # 1. FIX: Await the tools directly from the multi-server adapter
    lc_tools = await config.mcp_client.get_tools()

    # 2. Convert LangChain tool wrappers into OpenAI tools format for the NVIDIA client
    from langchain_core.utils.function_calling import convert_to_openai_tool
    openai_tools = [convert_to_openai_tool(t) for t in lc_tools]

    prompt = (
        "You are a calculation router bridging to a SymPy mathematical engine.\n"
        "You must pass complex calculations or algebraic strings to your calculation engine tool.\n"
        "CRITICAL SYNTAX RULES:\n"
        "- Use standard Python operators (e.g., use x**2 for squaring, not x^2).\n"
        "- For derivatives, use 'sp.diff(expression, x)'.\n"
        "- For equations or finding roots, use 'sp.solve(expression, x)'.\n\n"
        f"User Query: {question}"
    )

    # 3. Request inference with converted tools structure bound
    response = await config.llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=openai_tools,
        timeout=60
    )

    message = response.choices[0].message

    # 4. Execute tool call if requested by Llama 3.1
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"[Math Agent] Model invoking MCP tool: {tool_name} with args: {tool_args}")

        # Cross-reference the selected function name against your LangChain tools pool
        target_tool = next(t for t in lc_tools if t.name == tool_name)
        execution_output = await target_tool.ainvoke(tool_args)

        final_answer = str(execution_output)
        return {
            "answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
            "citations": ["Internal MCP MathEngine Subprocess"],
            "next_agent": "MATH"
        }

    ai_answer = message.content
    return {
        "answer": ai_answer,
        "messages": [AIMessage(content=ai_answer)],
        "citations": [],
        "next_agent": "MATH"
    }


async def node_rag_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running RAG Agent (NVIDIA)...")
    question = state["question"]
    rows = await retrieve(question)

    if not rows:
        err_msg = "I couldn't find matching info in your documents."
        return {
            "answer": err_msg,
            "messages": [AIMessage(content=err_msg)],
            "citations": [],
            "next_agent": "RAG"
        }

    context_parts = []
    citations = []
    for row in rows:
        citations.append(f"{row[0]} (Chunk {row[1]})")
        context_parts.append(f"SOURCE: {row[0]}\n\n{row[2]}")

    context_text = "\n\n".join(context_parts)
    prompt = f"Answer from context ONLY.\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"

    response = await config.llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    ai_answer = response.choices[0].message.content
    return {
        "answer": ai_answer,
        "messages": [AIMessage(content=ai_answer)],
        "citations": list(set(citations)),
        "next_agent": "RAG"
    }

async def node_tavily_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running Tavily Web Agent (NVIDIA)...")
    question = state["question"]

    if not config.tavily_client:
        err_msg = "Tavily Web Agent could not run because TAVILY_API_KEY is unconfigured."
        return {
            "answer": err_msg,
            "messages": [AIMessage(content=err_msg)],
            "citations": [],
            "next_agent": "TAVILY"
        }

    try:
        # Offload sync client engine call to threadpool loop
        search_result = await asyncio.to_thread(config.tavily_client.search, query=question, max_results=3)
        context_parts = [f"URL: {r['url']}\nContent: {r['content']}" for r in search_result.get("results", [])]
        citations = [r['url'] for r in search_result.get("results", [])]

        context_text = "\n\n".join(context_parts)
        prompt = f"Answer using the web data below concisely.\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"

        response = await config.llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        ai_answer = response.choices[0].message.content
        return {
            "answer": ai_answer,
            "messages": [AIMessage(content=ai_answer)],
            "citations": citations,
            "next_agent": "TAVILY"
        }
    except Exception as e:
        err_ex = f"Tavily API Runtime Exception: {str(e)}"
        return {
            "answer": err_ex,
            "messages": [AIMessage(content=err_ex)],
            "citations": [],
            "next_agent": "TAVILY"
        }

async def node_github_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running GitHub Agent (NVIDIA)...")
    question = state["question"]
    critique = state.get("system_critique", "")

    if not config.github_client:
        err_msg = "Missing GITHUB_TOKEN environment variable."
        return {
            "answer": err_msg,
            "messages": [AIMessage(content=err_msg)],
            "citations": [],
            "next_agent": "GITHUB"
        }

    try:
        extractor = (
            "Extract 'owner/repo' format from query. Return ONLY that string. "
            "If the query is general or doesn't mention a specific repo, reply with 'NONE'.\n"
            f"Query: {question}"
        )

        repo_res = await config.llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": extractor}]
        )
        repo_name = repo_res.choices[0].message.content.strip() if repo_res.choices[0].message.content else "NONE"

        if "NONE" in repo_name or "/" not in repo_name:
            # Wrap PyGithub blocking API steps inside a helper method for thread scheduling
            def _get_user_repos():
                user = config.github_client.get_user()
                repos = user.get_repos()
                sample = [r.name for idx, r in enumerate(repos) if idx < 5]
                return user.login, repos.totalCount, sample, user.html_url

            login, total_repo_count, sample_repos, html_url = await asyncio.to_thread(_get_user_repos)

            prompt = f"""
            You are interacting with a user's authenticated GitHub account data.
            Account Owner Login Name: {login}
            Total Repositories on Account: {total_repo_count}
            Sample accessible repo names: {', '.join(sample_repos)}

            USER QUESTION: {question}
            PREVIOUS EVALUATOR CRITIQUE (IF ANY): {critique}

            Answer the user's question accurately. Make sure to adhere to any formatting instructions given in the question!
            """

            response = await config.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_answer = response.choices[0].message.content
            return {
                "answer": ai_answer,
                "messages": [AIMessage(content=ai_answer)],
                "citations": [html_url],
                "next_agent": "GITHUB"
            }

        # Offload single repository extraction logic
        def _get_repo_details(name):
            r = config.github_client.get_repo(name)
            issues = [f"#{i.number}: {i.title}" for idx, i in enumerate(r.get_issues(state="open")) if idx < 3]
            return r.full_name, r.description, issues, r.html_url

        full_name, description, issues, repo_url = await asyncio.to_thread(_get_repo_details, repo_name)
        issue_text = "\n".join(issues) if issues else "No open issues found."

        ctx = f"Repo: {full_name}\nDesc: {description}\nOpen Issues:\n{issue_text}"
        prompt = f"Answer using this repository context.\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{question}\nCRITIQUE: {critique}"

        response = await config.llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_answer = response.choices[0].message.content
        return {
            "answer": ai_answer,
            "messages": [AIMessage(content=ai_answer)],
            "citations": [repo_url],
            "next_agent": "GITHUB"
        }

    except Exception as e:
        err_ex = f"GitHub Agent caught an execution error: {str(e)}"
        return {
            "answer": err_ex,
            "messages": [AIMessage(content=err_ex)],
            "citations": [],
            "next_agent": "GITHUB"
        }

async def node_summary_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running Summary & Chat History Agent...")
    question = state["question"]

    history_lines = []
    for msg in state.get("messages", []):
        sender = "User" if msg.type == "human" else "Assistant"
        history_lines.append(f"{sender}: {msg.content}")
    history_str = "\n".join(history_lines)

    prompt = f"""You are an internal operations assistant with clear visibility into the ongoing conversation thread history.
    The user is asking you for a summary or has a question regarding the past sequence of interactions in this specific session.

    CURRENT CONVERSATION HISTORY LOG:
    {history_str}

    USER TASK:
    {question}

    Fulfill their request explicitly based on the logs above.
    """

    response = await config.llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    ai_summary = response.choices[0].message.content
    return {
        "answer": ai_summary,
        "messages": [AIMessage(content=ai_summary)],
        "citations": [],
        "next_agent": "SUMMARY_AGENT"
    }

async def node_orchestrator(state: AgentState) -> dict:
    print(f"\n[Orchestrator] Reviewing query: '{state['question']}'")
    history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-5:]])

    prompt = f"""You are a master routing supervisor and query rewriter.

    RECENT CONVERSATION HISTORY FOR CONTEXT:
    {history}

    ROUTING RULES:
    - Choose 'MATH' for computational problems, algebra, symbols, or equations.
    - Choose 'RAG' for internal files, uploaded documents, universities.
    - Choose 'GITHUB' for repository details, git issues, open PRs.
    - Choose 'TAVILY' for recent events, search engine lookups, live internet news.
    - Choose 'SUMMARY_AGENT' for summaries of this chat session.

    TASK:
    1. Select the correct agent.
    2. Analyze the Latest User Query. If it uses pronouns ("he", "it", "that position") or relies on past context, rewrite it into a completely standalone, explicit question. If it's already standalone, leave it as is.

    Provide your output strictly in valid JSON matching this schema:
    {{
        "chosen_agent": "MATH" |"RAG" | "GITHUB" | "TAVILY" | "SUMMARY_AGENT",
        "standalone_question": "The rewritten explicit question string",
        "reasoning": "string"
    }}

    Latest User Query: "{state['question']}"
    """

    response = await config.llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_content = response.choices[0].message.content

    if raw_content.startswith("```"):
        raw_content = raw_content.strip("```json").strip("```").strip()

    try:
        decision = json.loads(raw_content)
        agent = decision.get("chosen_agent", "TAVILY").upper()
        optimized_question = decision.get("standalone_question", state["question"])
    except Exception as parse_err:
        print(f"[Orchestrator Fallback] Parsing failed: {parse_err}. Raw content: {raw_content}")
        # Intelligent structural fallback
        if "MATH" in raw_content or "x**" in state["question"]:
            agent = "MATH"
        else:
            agent = "TAVILY"
        optimized_question = state["question"]

    print(f"[Orchestrator Decision] Route to: {agent}")
    print(f"[Orchestrator Rewriter] Optimized Query: '{optimized_question}'")
    return {"next_agent": agent, "question": optimized_question}

async def route_decision(state: AgentState) -> str:
    return state["next_agent"]

# =====================================================
# EVALUATION AGENT
# =====================================================

async def node_evaluator(state: AgentState) -> dict:
    print("\n[Evaluator] -> Judging with Gemini...")
    question = state["question"]
    answer = state["answer"]
    retry_count = state.get("retry_count", 0)
    current_route = state.get("next_agent", "TAVILY")

    if retry_count >= 2:
        print("[Evaluator Decision] Max retries reached. Handing over to HUMAN INTERVENTION.")
        return {"next_agent": "INTERVENT"}

    eval_prompt = f"""You are an objective AI quality judge...
    QUESTION: "{question}"
    ANSWER: "{answer}"
    """

    try:
        # Wrap the generative sync implementation safely into an external execution block
        def _call_gemini():
            return config.gemini.generate_content(
                eval_prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": EvalSchema
                }
            )
        response = await asyncio.to_thread(_call_gemini)
        result = json.loads(response.text)
        passed = result.get("passed", True)
        print(f"[Gemini Evaluator] Passed: {passed} | Reason: {result.get('reasoning')}")

        if passed:
            return {"next_agent": "ACCEPT"}
        else:
            return {
                "next_agent": current_route,
                "retry_count": retry_count + 1,
                "system_critique": result.get("reasoning")
            }

    except Exception as e:
        print(f"[Evaluator Error] Gemini failed: {e}. Handing over to HUMAN INTERVENTION.")
        return {"next_agent": "INTERVENT"}

def evaluate_decision(state: AgentState) -> str:
    return state["next_agent"]

# =====================================================
# HITL (Human-In-The-Loop)
# =====================================================
async def node_human_intervention(state: AgentState):
    print(f"\n[HITL State] Yielding control to state store for Thread: {state.get('question')}")

    human_response = interrupt({
        "question": state["question"],
        "generated_answer": state["answer"],
        "critique": state.get("system_critique", "")
    })

    return {
        "answer": human_response,
        "messages": [AIMessage(content=human_response)],
        "next_agent": "ACCEPT"
    }
