from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
import asyncio
import sys

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": ["mathserver.py"],
                "transport": "stdio"
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )
    chat = ChatOllama(model='llama3.2:latest', temperature=0)
    tools = await client.get_tools()
    agent = create_react_agent(chat, tools=tools)
    response = await agent.ainvoke({"messages": [("user", "What is 3+45*45+100 and tell me the weather in karnataka?")]})
    print(response['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())