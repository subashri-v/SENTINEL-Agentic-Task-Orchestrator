# math_server.py
from mcp.server.fastmcp import FastMCP
import re
import sympy as sp
import math

mcp = FastMCP("MathEngine")

@mcp.tool()
def calculate_expression(expression: str) -> str:
    """
    Evaluates arithmetic computations, variable simplifications, or calculus expressions.
    Input should be a valid Python/SymPy syntax string using 'x' as a variable.
    Examples: '2**10 * 5.5', 'sp.solve(3*x**2 + 2*x - 5, x)', 'sp.diff(x**2, x)'
    """
    try:
        # Normalize common string mismatches from the LLM
        # (?<![\w.]) skips names that are already qualified, so "sp.solve" is not rewritten to "sp.sp.solve"
        expression = re.sub(r"(?<![\w.])derivative\b", "sp.diff", expression)
        expression = re.sub(r"(?<![\w.])solve\b", "sp.solve", expression)

        if "x" in expression or "sp." in expression:
            # Provide an explicit, safe math evaluation context
            context = {
                "sp": sp, 
                "math": math,
                "x": sp.Symbol('x')
            }
            # Remove dangerous builtins
            result = eval(expression, {"__builtins__": None}, context)
            return f"Symbolic Solution: {str(result)}"
        else:
            return f"Numerical Result: {str(eval(expression, {'__builtins__': None}, {'math': math}))}"
            
    except Exception as e:
        return f"Calculation Error: {str(e)}. Please check your SymPy syntax structure."

if __name__ == "__main__":
    mcp.run(transport="stdio")