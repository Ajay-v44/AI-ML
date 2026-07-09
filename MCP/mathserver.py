from fastmcp import FastMCP
import math

mcp=FastMCP(
    name='mathserver',
    # description='Math MCP Server',
    version='1.0.0'
)

@mcp.tool()
def add(a: int, b: int)->int:
    """Add two numbers"""
    return a+b

@mcp.tool()
def sub(a: int, b: int)->int:
    """Subtract two numbers"""
    return a-b

@mcp.tool()
def mul(a: int, b: int)->int:
    """Multiply two numbers"""
    return a*b

@mcp.tool()
def div(a: int, b: int)->int:
    """Divide two numbers"""
    return a/b

@mcp.tool()
def power(a: int, b: int)->int:
    """Power of a number"""
    return a**b

@mcp.tool()
def mod(a: int, b: int)->int:
    """Modulo of two numbers"""
    return a%b

@mcp.tool()
def fact(a: int)->int:
    """Factorial of a number"""
    return math.factorial(a)

if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )