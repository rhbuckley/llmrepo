import pytest
from llmrepo.tools import BaseTool, ToolParameter

class ValidationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="validation_tool",
            description="A tool for testing parameter validation",
            parameters={
                "string_param": ToolParameter(
                    name="string_param",
                    type="str",
                    description="A string parameter",
                    required=True
                ),
                "int_param": ToolParameter(
                    name="int_param",
                    type="int",
                    description="An integer parameter",
                    required=False,
                    default=42
                ),
                "float_param": ToolParameter(
                    name="float_param",
                    type="float",
                    description="A float parameter",
                    required=True
                ),
                "bool_param": ToolParameter(
                    name="bool_param",
                    type="bool",
                    description="A boolean parameter",
                    required=False,
                    default=False
                ),
                "list_param": ToolParameter(
                    name="list_param",
                    type="list",
                    description="A list parameter",
                    required=False,
                    default=[]
                )
            }
        )
    
    def invoke(self, string_param: str, float_param: float, 
               int_param: int = 42, bool_param: bool = False,
               list_param: list = None) -> dict:
        return {
            "string": string_param,
            "int": int_param,
            "float": float_param,
            "bool": bool_param,
            "list": list_param or []
        }

@pytest.fixture
def validation_tool():
    return ValidationTool()

def test_valid_parameters(validation_tool):
    """Test that valid parameters are accepted"""
    result = validation_tool.invoke(
        string_param="test",
        float_param=1.23,
        int_param=10,
        bool_param=True,
        list_param=[1, 2, 3]
    )
    
    assert result["string"] == "test"
    assert result["float"] == 1.23
    assert result["int"] == 10
    assert result["bool"] is True
    assert result["list"] == [1, 2, 3]

def test_default_parameters(validation_tool):
    """Test that default parameters are used when not provided"""
    result = validation_tool.invoke(
        string_param="test",
        float_param=1.0
    )
    
    assert result["int"] == 42  # default value
    assert result["bool"] is False  # default value
    assert result["list"] == []  # default value

def test_type_validation(validation_tool):
    """Test that type validation works for each parameter type"""
    
    # Test string parameter
    with pytest.raises(TypeError, match="must be of type str"):
        validation_tool.invoke(string_param=123, float_param=1.0)
    
    # Test int parameter
    with pytest.raises(TypeError, match="must be of type int"):
        validation_tool.invoke(string_param="test", float_param=1.0, int_param="not_an_int")
    
    # Test float parameter
    with pytest.raises(TypeError, match="must be of type float"):
        validation_tool.invoke(string_param="test", float_param="not_a_float")
    
    # Test bool parameter
    with pytest.raises(TypeError, match="must be of type bool"):
        validation_tool.invoke(string_param="test", float_param=1.0, bool_param="not_a_bool")
    
    # Test list parameter
    with pytest.raises(TypeError, match="must be of type list"):
        validation_tool.invoke(string_param="test", float_param=1.0, list_param="not_a_list")

def test_required_parameters(validation_tool):
    """Test that required parameters are enforced"""
    
    # Test missing required string parameter
    with pytest.raises(ValueError, match="Missing required parameter: string_param"):
        validation_tool.invoke(float_param=1.0)
    
    # Test missing required float parameter
    with pytest.raises(ValueError, match="Missing required parameter: float_param"):
        validation_tool.invoke(string_param="test")

def test_unexpected_parameters(validation_tool):
    """Test that unexpected parameters are rejected"""
    with pytest.raises(ValueError, match="Unexpected parameters"):
        validation_tool.invoke(
            string_param="test",
            float_param=1.0,
            unexpected_param="should fail"
        )

def test_none_values(validation_tool):
    """Test handling of None values"""
    
    # None should not be accepted for required parameters
    with pytest.raises(TypeError, match="must be of type str"):
        validation_tool.invoke(string_param=None, float_param=1.0)
    
    # None should be converted to default for optional parameters
    result = validation_tool.invoke(
        string_param="test",
        float_param=1.0,
        int_param=None,
        list_param=None
    )
    assert result["int"] == 42  # default value
    assert result["list"] == []  # default value

def test_type_coercion(validation_tool):
    """Test that no implicit type coercion occurs"""
    
    # Integer should not be coerced to float
    with pytest.raises(TypeError, match="must be of type float"):
        validation_tool.invoke(string_param="test", float_param=1)
    
    # Float should not be coerced to int
    with pytest.raises(TypeError, match="must be of type int"):
        validation_tool.invoke(string_param="test", float_param=1.0, int_param=1.0)
    
    # String representations should not be coerced
    with pytest.raises(TypeError, match="must be of type int"):
        validation_tool.invoke(string_param="test", float_param=1.0, int_param="42") 