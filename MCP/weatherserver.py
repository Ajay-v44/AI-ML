from fastmcp import FastMCP
import math

mcp=FastMCP(
    name='weatherserver',
    # description='Weather MCP Server',
    version='1.0.0',
)

@mcp.tool()
def get_weather(city: str)->str:
    """Get weather of a city"""
    return f"Weather in {city} is sunny"

if __name__ == "__main__":
    mcp.run(
        transport="http"
    )
