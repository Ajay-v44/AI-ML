import os
import argparse
import sys
import threading
import time
import requests
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import uvicorn

load_dotenv()

# Reconfigure stdout to UTF-8 to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 1. Set your specific environment variables
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
os.environ["OPENAI_CHAT_MODEL"] = os.getenv("OPENAI_CHAT_MODEL")
os.environ["OPENAI_EMBED_MODEL"] = os.getenv("OPENAI_EMBED_MODEL")
os.environ["OPENAI_TEMPERATURE"] = os.getenv("OPENAI_TEMPERATURE")
os.environ["OPENAI_MAX_TOKENS"] = os.getenv("OPENAI_MAX_TOKENS")

# 2. Initialize the Kimi LLM
llm = ChatOpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    model=os.environ["OPENAI_CHAT_MODEL"],
    temperature=float(os.environ["OPENAI_TEMPERATURE"]),
    max_tokens=int(os.environ["OPENAI_MAX_TOKENS"])
)

# 3. Define the Agent Card Schema
class AgentCard(BaseModel):
    agent_name: str
    description: str
    capabilities: list[str]

# 4. Build the Movie Agent (Worker)
@tool
def fetch_movies(genre: str) -> str:
    """Fetches movies from a database."""
    print(f"\n[Tool] Movie Agent fetching: {genre}")
    return f"Found top {genre} movies: Inception, The Matrix, Interstellar."

# <-- Updated v1.0 syntax: create_agent, model=, system_prompt=
movie_agent = create_agent(
    model=llm, 
    tools=[fetch_movies], 
    system_prompt="You are a cinematic expert. Recommend movies and explain why they are good."
)

# 5. Define HTTP endpoints registry instead of direct references
AGENT_URL_REGISTRY = {
    "MovieSpecialist": os.getenv("AGENT_MOVIE_SPECIALIST_URL", "http://127.0.0.1:8001/chat")
}

agent_cards = [
    AgentCard(
        agent_name="MovieSpecialist",
        description="An expert in cinema that can recommend movies based on genres, moods, or actors.",
        capabilities=["fetch_movies"]
    )
]

# 6. Build the Discovery and Dispatch Tools
@tool
def read_agent_cards() -> str:
    """Reads the profiles of all available specialized agents."""
    print("\n[A2A] User Agent reading registry...")
    card_text = []
    for card in agent_cards:
        card_text.append(f"Agent: {card.agent_name} | Desc: {card.description} | Tools: {', '.join(card.capabilities)}")
    return "\n".join(card_text)

@tool
def dispatch_to_agent(agent_name: str, instructions: str) -> str:
    """Sends instructions to a specific agent by name over HTTP and returns their response."""
    if agent_name not in AGENT_URL_REGISTRY:
        return f"Error: No agent found with name {agent_name} in HTTP registry."
    
    endpoint = AGENT_URL_REGISTRY[agent_name]
    print(f"\n[A2A HTTP Dispatcher] Dispatching to remote agent -> {agent_name} at {endpoint}")
    print(f"[A2A HTTP Dispatcher] Payload: {instructions}")
    
    try:
        payload = {"instructions": instructions}
        response = requests.post(endpoint, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response content returned.")
          
        return f"Error from remote agent {agent_name} (HTTP {response.status_code}): {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Network Error contacting agent {agent_name}: {e}"

# 7. Build the User Agent (Supervisor)
user_agent_prompt = (
    "You are the central User Agent. When a user asks a question:\n"
    "1. Use `read_agent_cards` to discover available specialized agents.\n"
    "2. Determine which agent is best suited for the task based on their description.\n"
    "3. Use `dispatch_to_agent` to send a clear instruction to the chosen agent.\n"
    "4. Return the agent's findings to the user."
)

# <-- Updated v1.0 syntax: create_agent, model=, system_prompt=
user_agent = create_agent(
    model=llm, 
    tools=[read_agent_cards, dispatch_to_agent], 
    system_prompt=user_agent_prompt
)

# 8. Define FastAPI worker app endpoints
app = FastAPI(title="Movie Specialist Agent Service")

class ChatRequest(BaseModel):
    instructions: str

class ChatResponse(BaseModel):
    agent_name: str
    response: str
    status: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        print(f"\n[Worker Service] Received remote call: '{request.instructions}'")
        response = movie_agent.invoke({"messages": [("user", request.instructions)]})
        agent_output = response["messages"][-1].content
        print(f"[Worker Service] Agent responded successfully.")
        return ChatResponse(
            agent_name="MovieSpecialist",
            response=agent_output,
            status="success"
        )
    except Exception as e:
        print(f"[Worker Service] Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 9. CLI Execution routines
def run_supervisor():
    print("Initializing A2A central User Agent (Supervisor)...")
    print(f"Registry configured with: {AGENT_URL_REGISTRY}")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            response = user_agent.invoke({
                "messages": [("user", user_input)]
            })
            
            print(f"\nFinal Output:\n{response['messages'][-1].content}")
        except KeyboardInterrupt:
            print("\nExiting Supervisor CLI...")
            break
        except Exception as e:
            print(f"\nError running supervisor: {e}")

def run_worker():
    url = AGENT_URL_REGISTRY["MovieSpecialist"]
    host = "127.0.0.1"
    port = 8001
    try:
        parts = url.replace("http://", "").replace("https://", "").split("/")[0].split(":")
        if len(parts) == 2:
            host = parts[0]
            port = int(parts[1])
    except Exception:
        pass
        
    print(f"Starting Movie Specialist Worker Service on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="warning")

def run_demo():
    url = AGENT_URL_REGISTRY["MovieSpecialist"]
    host = "127.0.0.1"
    port = 8001
    try:
        parts = url.replace("http://", "").replace("https://", "").split("/")[0].split(":")
        if len(parts) == 2:
            host = parts[0]
            port = int(parts[1])
    except Exception:
        pass

    print(f"Starting background Movie Worker Service on {host}:{port}...")
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=host, port=port, log_level="warning"),
        daemon=True
    )
    server_thread.start()
    
    # Wait for the FastAPI server to initialize
    time.sleep(2)
    print("Worker Service is active. Initializing Supervisor CLI...")
    run_supervisor()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed A2A Agent runner")
    parser.add_argument(
        "--mode", 
        choices=["demo", "worker", "supervisor"], 
        default="demo",
        help="Run mode: 'demo' (starts background worker + supervisor CLI), 'worker' (starts standalone worker REST API), or 'supervisor' (starts standalone supervisor CLI)"
    )
    args = parser.parse_args()

    if args.mode == "worker":
        run_worker()
    elif args.mode == "supervisor":
        run_supervisor()
    else:
        run_demo()