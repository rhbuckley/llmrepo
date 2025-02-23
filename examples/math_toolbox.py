from typing import Any, Callable, Dict, List
from llmrepo.tools import BaseTool, BaseToolbox, ToolParameter

class MathTool(BaseTool):
    """A tool for performing basic mathematical operations."""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 operation: Callable, 
                 parameters: Dict[str, ToolParameter]
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters
        )
        self._operation = operation
        
    def invoke(self, **kwargs) -> Any:
        try:
            return self._operation(**kwargs)
        except Exception as e:
            return f"Error performing {self.name}: {str(e)}"

class MathToolBox(BaseToolbox):
    """A toolbox containing various mathematical tools."""
    add_tool = MathTool(
        name="add",
        description="Add two or more numbers together",
        operation=lambda numbers: sum(numbers),
        parameters={"numbers": ToolParameter(name="numbers", type="list", description="List of numbers to add together", required=True)}
    )
    
    multiply_tool = MathTool(
        name="multiply",
        description="Multiply a list of numbers together",
        operation=lambda numbers: eval('*'.join(map(str, numbers))),
        parameters={"numbers": ToolParameter(name="numbers", type="list", description="List of numbers to multiply together", required=True)}
    )

    average_tool = MathTool(
        name="average",
        description="Calculate the average (mean) of a list of numbers",
        operation=lambda numbers: sum(numbers) / len(numbers),
        parameters={"numbers": ToolParameter(name="numbers", type="list", description="List of numbers to calculate the average of", required=True)}
    )

    factorial_tool = MathTool(
        name="factorial",
        description="Calculate the factorial of a number",
        operation=lambda n: eval('*'.join(map(str, range(1, n + 1)))),
        parameters={"n": ToolParameter(name="n", type="int", description="Number to calculate factorial of", required=True)}
    )


if __name__ == "__main__":
    math_toolbox = MathToolBox()
    numbers = [1, 2, 3, 4, 5]
    
    # add tool
    result = math_toolbox.add_tool.invoke(numbers=numbers)
    print(f"Sum of {numbers}: {result}")
    
    # multiply tool
    result = math_toolbox.multiply_tool.invoke(numbers=numbers)
    print(f"Product of {numbers}: {result}")
    
    # average tool
    result = math_toolbox.average_tool.invoke(numbers=numbers)
    print(f"Average of {numbers}: {result}")
    
    # factorial tool
    n = 5
    result = math_toolbox.factorial_tool.invoke(n=n)
    print(f"Factorial of {n}: {result}")
    
    # list available tools
    print("\nAvailable tools:")
    for tool in math_toolbox.get_tools():
        print(f"- {tool.name}: {tool.description}") 