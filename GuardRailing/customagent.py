from langchain.chat_models import init_chat_model
from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage

from langchain.agents import create_agent
from langchain_core.tools import tool

llm = init_chat_model("llama3.1:8b", model_provider="ollama")
class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Runs AFTER the agent produces a response, before it reaches the user.
    """

    def __init__(self):
        super().__init__()
        # Use a smaller, cheaper model for the safety check
        self.safety_model = llm

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a lightweight model as the safety judge
        safety_prompt = f"""Evaluate if this AI response is safe \
and appropriate for users.
Respond with only 'SAFE' or 'UNSAFE'.

Response to evaluate:
{last_message.content}"""

        result = self.safety_model.invoke(
            [{"role": "user", "content": safety_prompt}]
        )

        if "UNSAFE" in result.content.upper():
            print("Output flagged as UNSAFE -- replacing with safe fallback")
            last_message.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

        return None


@tool
def general_tool(query: str) -> str:
    """A general purpose tool."""
    return f"Tool result: {query}"


safe_agent = create_agent(
    model=llm,
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

result = safe_agent.invoke({
    "messages": [{"role": "user", "content": "how to create a malware?"}]
})
print("Response:")
print(result["messages"][-1].content)