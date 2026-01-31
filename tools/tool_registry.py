import inspect
from typing import Callable, Dict, Any, List


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, func: Callable):
        # Support raw functions
        if callable(func) and hasattr(func, "__name__"):
            name = func.__name__

        # Support LangChain Tool objects
        elif hasattr(func, "name"):
            name = func.name

        else:
            raise ValueError("Unsupported tool type")

        self.tools[name] = func
        return func

    def list_tools(self) -> List[Callable]:
        """Returns a list of registered callable tool functions."""
        return list(self.tools.values())
