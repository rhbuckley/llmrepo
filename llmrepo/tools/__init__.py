import asyncio
import logging
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, List, Optional, Union, Literal



class ToolEvent(str, Enum):
    """Events that can be hooked into for the tool lifecycle"""
    BEFORE_INVOKE = "invoke:before"
    AFTER_INVOKE = "invoke:after"
    BEFORE_AINVOKE = "ainvoke:before"
    AFTER_AINVOKE = "ainvoke:after"
    ERROR = "error"


# Create a type for valid event strings
ToolEventType = Literal[
    "invoke:before",
    "invoke:after",
    "ainvoke:before",
    "ainvoke:after",
    "error"
]


class ToolParameter(BaseModel):
    """
    This model represents a parameter for a tool. Parameters
    must be specified in the metadata of the tool, as otherwise,
    the LLM may not provide the correct parameters.

    Args:
        name: The name of the parameter.
        description: A description of the parameter.
        type: The type of the parameter.
        required: Whether the parameter is required.
        default: The default value of the parameter.
    """
    name: str = Field(..., description="The name of the parameter")
    type: str = Field(..., description="The type of the parameter")
    description: str = Field(..., description="A description of the parameter")
    required: bool = Field(default=True, description="Whether the parameter is required")
    default: Optional[Any] = Field(default=None, description="The default value of the parameter")


class ToolMetadata(BaseModel):
    """
    Base model for common metadata fields used in tools and other models.
    
    Attributes:
        name: The name of the model/tool.
        description: A detailed description of what the model/tool does.
        parameters: A dictionary defining the expected parameters and their specifications.
    """
    name: str = Field(..., description="The name of the model/tool")
    description: str = Field(..., min_length=1, description="A detailed description of what the model/tool does")
    parameters: Dict[str, ToolParameter] = Field(
        default_factory=dict,
        description="Dictionary of parameters this model/tool accepts"
    )


class ToolContext:
    """
    A class to manage tool context with support for both internal and shared state.
    
    Attributes:
        _internal_context: Dictionary storing tool-specific context
        _shared_context: Dictionary storing context shared across tools
    """
    
    def __init__(self, initial_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager with optional initial context.
        
        Args:
            initial_context: Optional dictionary to initialize internal context
        """
        self._internal_context: Dict[str, Any] = initial_context or {}
        self._shared_context: Dict[str, Any] = {}
    
    @property
    def context(self) -> Dict[str, Any]:
        """
        Gets a combined view of internal and shared context.
        Internal context takes precedence over shared context.
        """
        combined = self._shared_context.copy()
        combined.update(self._internal_context)
        return combined
    
    @context.setter
    def context(self, value: Dict[str, Any]) -> None:
        """
        Sets the shared context. This is typically called by the toolbox.
        """
        self._shared_context = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a context value, checking internal context first, then shared context.
        
        Args:
            key: The context key to look up
            default: Value to return if key is not found in either context
        """
        if key in self._internal_context:
            return self._internal_context[key]
        return self._shared_context.get(key, default)
    
    def set(self, key: str, value: Any, force_shared: bool = False) -> None:
        """
        Sets a context value, updating whichever context currently contains the key,
        or internal context if the key doesn't exist.
        
        Args:
            key: The context key to set
            value: The value to set
            force_shared: If True, always set in shared context
        """
        if force_shared:
            self._shared_context[key] = value
        elif key in self._shared_context:
            logging.debug(f"Updating shared context key: {key}")
            self._shared_context[key] = value
        elif key in self._internal_context:
            logging.debug(f"Updating internal context key: {key}")
            self._internal_context[key] = value
        else:
            logging.debug(f"Creating new internal context key: {key}")
            self._internal_context[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to context values."""
        if key in self._internal_context:
            return self._internal_context[key]
        if key in self._shared_context:
            return self._shared_context[key]
        raise KeyError(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of context values."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Support for 'in' operator."""
        return key in self._internal_context or key in self._shared_context


class BaseTool(ToolMetadata, ABC):
    """
    An abstract base class that represents a tool that can be used by an LLM.
    Tools must implement synchronous (invoke) execution methods, with optional
    async implementation (ainvoke). Each tool must specify its name, description,
    and expected parameters.

    Attributes:
        context: ToolContext instance managing internal and shared state
        _hooks: Internal dictionary of event hooks and their callbacks.
    """
    context: ToolContext = Field(default_factory=ToolContext)
    _hooks: Dict[ToolEvent, List[Callable]] = defaultdict(list)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs) -> None:
        """
        Initializes a new tool instance with context from kwargs.
        """
        super().__init__(**kwargs)
        if 'context' in kwargs:
            self.context = ToolContext(kwargs.get('context'))


    def on(self, event: Union[ToolEvent, ToolEventType], callback: Callable) -> None:
        """
        Registers a callback function for a specific event.

        Args:
            event: The event to hook into, either as ToolEvent enum or string
            callback: The function to call when the event occurs

        Raises:
            ValueError: If the event is not a valid ToolEvent
        """
        if isinstance(event, str):
            try:
                event = ToolEvent(event)
            except ValueError:
                valid_events = ", ".join(e.value for e in ToolEvent)
                raise ValueError(
                    f"Invalid event '{event}'. Valid events are: {valid_events}"
                )
        
        self._hooks[event].append(callback)

    def _validate_parameters(self, kwargs: Dict[str, Any]) -> None:
        """
        Validates that the provided parameters match the expected parameters.
        
        Args:
            kwargs: Dictionary of parameters to validate
            
        Raises:
            ValueError: If unexpected or missing required parameters are found
            TypeError: If parameter types don't match expected types
        """
        if not kwargs:
            return
            
        # check for unexpected parameters 
        unexpected_params = set(kwargs.keys()) - set(self.parameters.keys())
        if unexpected_params:
            raise ValueError(f"Unexpected parameters: {unexpected_params}")
        
        # define mapping of type strings to actual types
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set
        }
        
        # check required parameters and types
        for param_name, param_spec in self.parameters.items():
            if param_spec.required and param_name not in kwargs:
                if param_spec.default is None:
                    raise ValueError(f"Missing required parameter: {param_name}")
                kwargs[param_name] = param_spec.default
            
            # if parameter is provided, validate its type
            if param_name in kwargs:
                param_value = kwargs[param_name]
                
                # Handle None for optional parameters
                if param_value is None:
                    if param_spec.required:
                        raise TypeError(
                            f"Parameter '{param_name}' must be of type {param_spec.type}, "
                            f"got NoneType"
                        )
                    # For optional parameters, set to default if None
                    kwargs[param_name] = param_spec.default
                    continue
                
                type_str = param_spec.type.lower()
                if type_str not in type_map:
                    raise ValueError(f"Unsupported parameter type: {param_spec.type}")
                expected_type = type_map[type_str]
                if not isinstance(param_value, expected_type):
                    raise TypeError(
                        f"Parameter '{param_name}' must be of type {param_spec.type}, "
                        f"got {type(param_value).__name__}"
                    )

    def __getattribute__(self, name: str) -> Any:
        """
        Intercepts method calls to handle event triggering.
        """
        attr = super().__getattribute__(name)
        
        if not hasattr(attr, '__call__') or name not in ['invoke', 'ainvoke']:
            return attr
        
        events = {
            'before': {
                'invoke': ToolEvent.BEFORE_INVOKE,
                'ainvoke': ToolEvent.BEFORE_AINVOKE
            },
            'after': {
                'invoke': ToolEvent.AFTER_INVOKE,
                'ainvoke': ToolEvent.AFTER_AINVOKE
            },
        }

        if name == 'invoke':
            def wrapped(*args, **kwargs):
                # Synchronous event triggering for invoke
                for callback in self._hooks[events['before'][name]]:
                    callback(self, args=args, kwargs=kwargs)
                
                try:
                    self._validate_parameters(kwargs)
                    result = attr(*args, **kwargs)
                    
                    for callback in self._hooks[events['after'][name]]:
                        callback(self, result=result, args=args, kwargs=kwargs)
                    
                    return result
                except Exception as e:
                    for callback in self._hooks[ToolEvent.ERROR]:
                        callback(self, error=e, args=args, kwargs=kwargs)
                    raise
            return wrapped
        else:  # name == 'ainvoke'
            async def awrapped(*args, **kwargs):
                await self._trigger_event_async(
                    events['before'][name],
                    args=args,
                    kwargs=kwargs
                )
                try:
                    self._validate_parameters(kwargs)
                    result = await attr(*args, **kwargs)
                    
                    await self._trigger_event_async(
                        events['after'][name],
                        result=result,
                        args=args,
                        kwargs=kwargs
                    )
                    return result
                except Exception as e:
                    await self._trigger_event_async(
                        ToolEvent.ERROR,
                        error=e,
                        args=args,
                        kwargs=kwargs
                    )
                    raise
            return awrapped

    async def _trigger_event_async(self, event: ToolEvent, **kwargs) -> None:
        """
        Triggers all callbacks registered for a specific event asynchronously.

        Args:
            event: The event to trigger
            **kwargs: Arguments to pass to the callback functions
        """
        for callback in self._hooks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, **kwargs)
                else:
                    callback(self, **kwargs)
            except Exception as e:
                logging.error(f"Error in async {event} callback: {str(e)}")
                if event != ToolEvent.ERROR:
                    await self._trigger_event_async(
                        ToolEvent.ERROR,
                        error=e,
                        source_event=event,
                        **kwargs
                    )

    @abstractmethod
    def invoke(self, *args, **kwargs):
        """
        Executes the tool's functionality synchronously.
        Must be implemented by concrete tool classes.

        Args:
            *args: Variable positional arguments passed to the tool.
            **kwargs: Variable keyword arguments passed to the tool.

        Returns:
            The result of the tool's execution.
        """
        pass
    
    async def ainvoke(self, *args, **kwargs):
        """
        Executes the tool's functionality asynchronously.
        By default, falls back to synchronous implementation.

        Args:
            *args: Variable positional arguments passed to the tool.
            **kwargs: Variable keyword arguments passed to the tool.

        Returns:
            The result of the tool's execution.
        """
        logging.warning((
            f"Tool {self.name} has no async implementation"
            f", defaulting to sync implementation"
        ))
        return self.invoke(*args, **kwargs)
    
    def as_openai_tool(self) -> Dict[str, Any]:
        """
        Formats the tool's metadata into the OpenAI function calling format.
        This allows the tool to be used with OpenAI's function calling API.

        Returns:
            dict: A dictionary containing the tool's metadata in OpenAI's format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    name: param.model_dump(exclude_none=True)
                    for name, param in self.parameters.items()
                },
            },
        }


class BaseToolbox(ABC):
    """
    An abstract base class that represents a collection of related tools.
    The toolbox provides methods to access and format all available tools.

    Attributes:
        context: A dictionary to store any contextual information shared across tools.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a new toolbox instance with context from kwargs.
        The context can be used to share state or configuration across multiple tools.
        """
        self.context = kwargs.get('context', {})
        self._inject_context_to_tools()
    
    def _inject_context_to_tools(self) -> None:
        """
        Injects the toolbox's context into all tools managed by this toolbox.
        All tools share the same context reference to allow state sharing.
        """
        for tool in self.get_tools():
            # Share the same context reference across all tools
            tool.context._shared_context = self.context
    
    def get_tools(self) -> List[BaseTool]:
        """
        Discovers all tool instances that are defined as class attributes
        of the toolbox. This allows toolboxes to define their tools as
        class attributes and have them automatically collected.

        Example:
            class SearchToolbox(BaseToolbox):
                google_search = GoogleSearchTool()
                bing_search = BingSearchTool()

        Returns:
            List[BaseTool]: A list of all tool instances defined in the toolbox.
        """
        tools = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, BaseTool):
                tools.append(attr)
        return tools
    
    def on(self, event: Union[ToolEvent, ToolEventType], callback: Callable) -> None:
        """
        Registers a callback function for a specific event on all tools.
        """
        if isinstance(event, str):
            event = ToolEvent(event)
        
        for tool in self.get_tools():
            tool.on(event, callback)
    
    def as_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Formats all tools in the toolbox into OpenAI's function calling format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing each tool's
            metadata in OpenAI's format.
        """
        return [tool.as_openai_tool() for tool in self.get_tools()]
    
    @classmethod
    def from_tools(cls, tools: List[BaseTool], context: Optional[Dict[str, Any]] = None, **kwargs) -> "BaseToolbox":
        """
        Creates a toolbox from a list of tools.
        """
        return cls(tools=tools, context=context, **kwargs)


