from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Define the tool
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

tools = [add]

# 2. Initialize the model 
# Note: Ensure your local Ollama instance supports tool calling for this model
model = ChatOllama(model="llama3.2:latest", temperature=0.7)

# 3. Use an InMemorySaver for state persistence
memory = InMemorySaver()

# 4. Create the agent and interrupt *before* the action node executes tools
agent = create_react_agent(
    model, 
    tools=tools, 
    checkpointer=memory,
    interrupt_before=["tools"]  # This pauses the agent right before executing ANY tool
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

# 5. First invocation (The agent will decide to call the tool, then pause)
print("--- Initial Invocation ---")
for chunk in agent.stream({"messages": [("user", "what is 1+1?")]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# Check the state to see if it's interrupted
state = agent.get_state(config)
print(f"\nNext step to execute: {state.next}")

# 6. Human Approval Logic
# If state.next is 'tools', it's waiting for your approval!
if state.next and state.next[0] == "tools":
    user_decision = input("Do you approve this tool call? (yes/no): ").strip().lower()
    
    if user_decision == "yes":
        print("\n--- Resuming Agent ---")
        # Resume by passing None for inputs, keeping the original stream going
        for chunk in agent.stream(None, config, stream_mode="values"):
            chunk["messages"][-1].pretty_print()
    else:
        print("\nTool call rejected by human.")
        # Optional: update state with a rejection message or stop execution