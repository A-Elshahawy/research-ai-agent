from langchain.agents import Tool


class ResearchTool:
    """Base class for research tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, input_text: str) -> str:
        """Run the tool with the given input."""
        raise NotImplementedError("Subclasses must implement run()")

    def to_langchain_tool(self) -> Tool:
        """Convert to a LangChain Tool object."""
        return Tool(name=self.name, description=self.description, func=self.run)
