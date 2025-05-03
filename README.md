# Intelligent AI Agent for Automated Research

This repository implements a modular AI agent system with planning and execution capabilities for automated research tasks. The system uses LLM-powered reasoning with ReAct prompting (Reasoning + Acting) to perform comprehensive research on any topic.

## Project Overview

The Research AI Agent is designed to autonomously gather, analyze, and synthesize information on a wide range of topics. The system follows a ReAct (Reasoning + Acting) paradigm that combines:

1. **Reasoning** : Using LLMs to plan research steps and analyze information
2. **Acting** : Executing tools to search, scrape, analyze, and generate content
3. **Memory** : Maintaining context and prior research findings

This approach allows the agent to break down complex research tasks, use appropriate tools for data gathering and analysis, and synthesize coherent research summaries.

## Features

* Modular agent architecture with specialized tools
* ReAct (Reasoning + Acting) framework for planning and execution
* Context-aware memory system for research continuity
* Web search and scraping capabilities
* Data analysis tools with visualization descriptions
* Content generation for research summaries
* Conversation history tracking

## Components

### Core Components

* **ResearchAgent** : The main agent class coordinating the research process
* **AgentMemory** : Memory system for conversation and research findings
* **ReActOutputParser** : Parser for the ReAct format outputs

### Tool Modules

* **WebSearchTool** : Simulated web search functionality
* **WebScraperTool** : Website content extraction
* **DataAnalysisTool** : Numerical data analysis with statistical insights
* **ContentGeneratorTool** : Research summary and content generation

## Requirements

* Python 3.11+
* langchain
* openai
* beautifulsoup4
* requests
* pandas
* matplotlib
* numpy
* dotenv

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/A-Elshahawy/research-ai-agent.git
   cd research-ai-agent
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### 1. Basic Agent Usage

```python
from research_agent import ResearchAgent

# Initialize the research agent
agent = ResearchAgent(llm_model="gpt-3.5-turbo", temperature=0)

# Run the agent on a research query
query = "What are the latest advancements in transformer models for NLP?"
response = agent.run(query)

# Print the research results
print(response)
```

### 2. Using the Agent with Custom Tools

You can extend the agent with custom tools:

```python
from research_agent import ResearchAgent, ResearchTool

# Define a custom research tool
class CustomDatabaseTool(ResearchTool):
    def __init__(self):
        super().__init__(
            name="database_search",
            description="Search a specialized research database. Input should be a search query."
        )
  
    def run(self, query: str) -> str:
        # Implement your database search logic here
        return f"Database results for: {query}"

# Initialize the agent with custom tools
agent = ResearchAgent(llm_model="gpt-3.5-turbo", temperature=0)
agent.tools.append(CustomDatabaseTool())  # Add custom tool
agent.langchain_tools = [tool.to_langchain_tool() for tool in agent.tools]  # Update tools
agent.agent_executor = agent._setup_agent()  # Reinitialize the agent executor

# Use the agent
response = agent.run("Research quantum computing algorithms")
print(response)
```

### 3. Continuous Research Session

The agent maintains memory of previous research, allowing for continuous sessions:

```python
# Initialize the agent
agent = ResearchAgent()

# First research query
response1 = agent.run("What are the basics of transformer models?")
print("FIRST QUERY RESULTS:")
print(response1)

# Follow-up question using previous context
response2 = agent.run("How do they handle long sequences?")
print("\nFOLLOW-UP QUERY RESULTS:")
print(response2)
```

### 4. Run Complete Example

To run the complete example demonstrating multiple research queries:

```bash
python research_agent.py
```

## Customizing the Agent

### Change the Language Model

```python
# Use GPT-4 for more advanced reasoning
agent = ResearchAgent(llm_model="gpt-4", temperature=0)

# Use Claude model if available
from langchain.chat_models import ChatAnthropic
claude_llm = ChatAnthropic(model="claude-2")
agent = ResearchAgent(llm=claude_llm)
```

### Modify Memory Parameters

```python
# Initialize agent with custom memory parameters
agent = ResearchAgent()
agent.memory.max_tokens = 8000  # Increase context window
```

### Implement Real Web Search

Replace the mock WebSearchTool with a real search API:

```python
class RealWebSearchTool(ResearchTool):
    def __init__(self, api_key):
        super().__init__(
            name="web_search",
            description="Search the web for information. Input should be a search query."
        )
        self.api_key = api_key
  
    def run(self, query: str) -> str:
        # Implement real search using Google Custom Search, Bing, or similar
        # Example with Google Custom Search:
        import requests
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": "your_search_engine_id",
            "q": query
        }
        response = requests.get(url, params=params)
        results = response.json()
    
        # Format results
        results_text = "Search results:\n"
        for i, item in enumerate(results.get("items", [])[:5]):
            results_text += f"{i+1}. {item.get('title')}\n   {item.get('snippet')}\n   URL: {item.get('link')}\n\n"
    
        return results_text
```

## Architecture Diagram

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│                 │     │               │     │                 │
│  User Query     │────▶│  ReAct Agent  │────▶│  Research Plan  
│                 │     │               │     │                 │
└─────────────────┘     └───────┬───────┘     └─────────┬───────┘
                                │                       │
                                ▼                       ▼
                        ┌───────────────┐     ┌─────────────────┐
                        │               │     │                 │
                        │  Agent Memory │◀────│  Tool Selection 
                        │               │     │                 │
                        └───────┬───────┘     └─────────┬───────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│                 │     │               │     │                 │
│  Research Report│◀────│  Synthesize   │◀────│   Tool Action   
│                 │     │               │     │                 │
└─────────────────┘     └───────────────┘     └─────────────────┘
```

## Extending the Agent

### Adding New Tools

1. Create a new tool class inheriting from `ResearchTool`
2. Implement the `run` method to define the tool's functionality
3. Add the tool to the agent's tool list

Example:

```python
from research_agent import ResearchTool

class PDFExtractorTool(ResearchTool):
    def __init__(self):
        super().__init__(
            name="pdf_extractor",
            description="Extract text from PDF files. Input should be a file path."
        )
  
    def run(self, file_path: str) -> str:
        import PyPDF2
    
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return f"Extracted {len(reader.pages)} pages from {file_path}:\n\n{text[:1000]}..."
        except Exception as e:
            return f"Error extracting PDF content: {str(e)}"
```

### Customizing ReAct Prompting

You can modify the ReAct template to customize agent behavior:

```python
# Custom ReAct template
custom_template = """You are a specialized research assistant for scientific literature.
You have access to the following tools:

{tools}

Use the following format:
Thought: consider what specific scientific information you need
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have the scientific information needed
Final Answer: provide a detailed scientific analysis with citations

Begin!

Previous conversation history:
{chat_history}

{memory_context}

New research question: {input}
Thought:"""

# Apply the custom template
from langchain.prompts import PromptTemplate
agent = ResearchAgent()
agent.agent_executor = agent._setup_agent(template=custom_template)
```

## Limitations and Future Work

Current limitations and areas for future enhancement:

1. **Web Search** : Currently uses a simulated search; integration with real search APIs is recommended
2. **Citation Management** : Limited support for formal citation tracking
3. **Tool Verification** : No verification of tool outputs for factuality
4. **Memory Management** : Basic memory system without sophisticated retrieval mechanisms
5. **Multi-step Planning** : Limited to short-term planning; could benefit from hierarchical planning

Future work could focus on:

* Adding support for academic database integration
* Implementing structured citation management
* Developing specialized domain-specific research tools
* Enhancing the memory system with more sophisticated retrieval

## License

This project is licensed under the MIT License - see the LICENSE file for details.uv
