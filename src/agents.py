from typing import Any, Dict

from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.agents.agent import AgentOutputParser
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from .agent_mem import AgentMemory
from .tools import ContentGeneratorTool, DataAnalysisTool, WebScraperTool, WebSearchTool


class ResearchAgent:
    """AI agent for conducting automated research."""

    def __init__(self, llm_model="gpt-3.5-turbo", temperature=0):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)  # type: ignore
        self.memory = AgentMemory()

        # Initialize tools
        self.tools = [
            WebSearchTool(),
            WebScraperTool(),
            DataAnalysisTool(),
            ContentGeneratorTool(self.llm),
        ]

        # Convert to LangChain tools
        self.langchain_tools = [tool.to_langchain_tool() for tool in self.tools]

        # Set up the agent with ReAct prompting
        self.agent_executor = self._setup_agent()

    def _setup_agent(self) -> AgentExecutor:
        """Set up the agent with tools and ReAct prompting."""
        # ReAct prompt template
        react_template = """You are an intelligent research assistant that helps with gathering and analyzing information.
        You have access to the following tools:

        {tools}

        Use the following format:

        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the user's question

        Begin!

        Previous conversation history:
        {chat_history}

        {memory_context}

        New question: {input}
        Thought:"""

        prompt = PromptTemplate.from_template(
            template=react_template,
            partial_variables={
                "tools": "\n".join(
                    [
                        f"{tool.name}: {tool.description}"
                        for tool in self.langchain_tools
                    ]
                ),
                "tool_names": ", ".join([tool.name for tool in self.langchain_tools]),
            },
        )

        # Set up LLM chain for the agent
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Output parser for ReAct format
        output_parser = ReActOutputParser()

        # Create the agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["Observation:"],
            allowed_tools=[tool.name for tool in self.langchain_tools],  # type: ignore
        )

        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.langchain_tools,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
        )

        return agent_executor

    def run(self, query: str) -> str:
        """Run the research agent with a user query."""
        # Get memory context
        memory_context = self.memory.get_memory_context(query)

        # Get conversation history
        chat_history = self.memory.get_conversation_history()

        # Execute the agent
        response = self.agent_executor.run(
            input=query, chat_history=chat_history, memory_context=memory_context
        )

        # Update memory
        self.memory.add_message("human", query)
        self.memory.add_message("ai", response)

        # Add research finding to memory
        self.memory.add_research_item(
            {
                "title": f"Research on: {query[:50]}",
                "content": response[:500],  # Store a summary in memory
                "full_content": response,
                "query": query,
            }
        )

        return response


class ReActOutputParser(AgentOutputParser):
    """Parser for ReAct format agent outputs."""

    def parse(self, llm_output: str) -> Dict[str, Any]:
        """Parse the LLM output into an action and action input."""
        # Check if the output contains a final answer
        if "Final Answer:" in llm_output:
            # Extract the final answer
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            return {"output": final_answer}

        # Extract the action and action input
        action_match = None
        action_input_match = None

        # Look for Action: and Action Input:
        for line in llm_output.split("\n"):
            if line.startswith("Action:"):
                action_match = line[len("Action:") :].strip()
            elif line.startswith("Action Input:"):
                action_input_match = line[len("Action Input:") :].strip()

        # Validate action and action input
        if not action_match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        return {"tool": action_match, "tool_input": action_input_match or ""}
