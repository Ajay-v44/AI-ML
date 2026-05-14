import uuid
import ollama
from langchain.chat_models import init_chat_model
from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, hook_config
)
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware
)
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage

OFF_TOPIC_REFUSAL = (
    "🚫 I'm a healthcare assistant and can only help with medical questions, "
    "symptoms, appointments, and health information. "
    "Please ask me something health-related."
)

def _is_healthcare_related(text: str) -> bool:
    """
    Uses the LLM as a binary domain classifier.
    Returns True if the input is related to healthcare/medical topics.
    """
    prompt = f"""You are a strict domain classifier for a healthcare assistant.

Decide whether the user message is related to healthcare, medicine, symptoms,
medications, appointments, mental health, nutrition, or general wellness.

Respond ONLY with one word: MEDICAL or OFFTOPIC

User message: {text}"""
    try:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        verdict = response["message"]["content"].strip().upper()
        return "OFFTOPIC" not in verdict
    except Exception:
        return True  # Fail open — let the agent handle it


class HealthcareSafetyFilter(AgentMiddleware):
    """Block non-medical or harmful requests in a healthcare context."""

    # Layer A: fast keyword check (zero latency)
    HARD_BLOCKED = [
        "drug synthesis", "self-harm", "suicide method",
        "weapon", "hack", "bomb",
    ]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        # ── BUG FIX: check the LAST human message, not messages[0] ──
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return None

        content = last_human.content

        # Layer A — hard keyword block
        for topic in self.HARD_BLOCKED:
            if topic in content.lower():
                return {
                    "messages": [{"role": "assistant", "content": OFF_TOPIC_REFUSAL}],
                    "jump_to": "end",
                }

        # Layer B — LLM domain classifier
        if not _is_healthcare_related(content):
            return {
                "messages": [{"role": "assistant", "content": OFF_TOPIC_REFUSAL}],
                "jump_to": "end",
            }

        return None
class MedicalOutputValidator(AgentMiddleware):
    """Ensure all responses include appropriate medical disclaimers."""

    DISCLAIMER = (
        "\n\nThis is general health information, not medical advice. "
        "Please consult a qualified healthcare professional."
    )

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Add disclaimer if not already present
        if "medical advice" not in last_message.content.lower():
            last_message.content += self.DISCLAIMER

        return None

@tool
def search_symptoms(symptoms: str) -> str:
    """Search for information about medical symptoms."""
    return (
        f"Symptom information for: {symptoms}. "
        "Please consult a doctor for diagnosis."
    )

@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment."""
    return (
        f"Appointment booked for {patient_name} "
        f"with Dr. {doctor} on {date}"
    )

@tool
def get_medication_info(medication: str) -> str:
    """Get information about a medication."""
    return (
        f"General info about {medication}. "
        "Always follow your doctor's prescription."
    )

llm = init_chat_model("llama3.1:8b", model_provider="ollama")
healthcare_bot = create_agent(
    model=llm,
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # Guardrail 1: Block harmful/off-topic requests
        HealthcareSafetyFilter(),

        # Guardrail 2: Redact patient PII from inputs
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware(
            "credit_card", strategy="mask", apply_to_input=True
        ),

        # Guardrail 3: Require approval before booking appointments
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_appointment": True,
                "search_symptoms": False,
                "get_medication_info": False,
            }
        ),

        # Guardrail 4: Add medical disclaimer to all outputs
        MedicalOutputValidator(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are a helpful healthcare assistant. "
        "You can search for symptoms, medication information, "
        "and help book appointments. Always be empathetic and "
        "remind users to consult a doctor for diagnosis."
    ),
)

print("Healthcare chatbot with full guardrail stack created!")



# Generate a fresh thread ID per session so InMemorySaver
# doesn't carry over stale conversation history between runs.
config_t1 = {"configurable": {"thread_id": f"session_{uuid.uuid4().hex[:8]}"}}

# result = healthcare_bot.invoke(
#     {"messages": [{"role": "user", "content": "What are symptoms of Type 2 Diabetes?"}]},
#     config=config_t1
# )
# print(result["messages"][-1].content)

# result = healthcare_bot.invoke({
#     "messages": [{
#         "role": "user",
#         "content": (
#             "My email is patient123@gmail.com. "
#             "What can I take for a headache?"
#         )
#     }]},
#     config=config_t1
# )
# print("=== PII Redaction Test ===")
# print(result["messages"][-1].content)

while True:
    try:
        inp=input("Ask: ")
        if inp=="exit":
            break
        result = healthcare_bot.invoke({
            "messages": [{"role": "user", "content": inp}]
        },
        config=config_t1
        )
        print(result["messages"][-1].content)
    except Exception as e:
        import traceback
        traceback.print_exc()


""""
Before a request goes to lmm we will filter and check if its off topic with a  small model.
If it is off topic we will not send it to the lmm. 
if not off topic then we will send it to the lmm.


After the lmm responds, we will check if the response is safe and appropriate for users. 
If the response is not safe or appropriate for users, we will replace it with a safe fallback.
if the response is safe and appropriate for users, we will return it to the user.

The guardrail is applied on input and output.
"""