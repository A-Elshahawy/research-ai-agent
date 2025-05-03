from .src.agents import ResearchAgent


def main():
    """Main function demonstrating the research agent."""
    # Initialize the research agent
    agent = ResearchAgent(llm_model="gpt-3.5-turbo", temperature=0)

    # Example research queries
    research_queries = [
        "What are the latest advancements in transformer models for NLP?",
        "How is machine learning being applied in climate science?",
        "Summarize the current state of quantum computing research",
    ]

    # Run the agent on each query
    for query in research_queries:
        print(f"\n\n=== RESEARCH QUERY: {query} ===\n")
        response = agent.run(query)
        print("\n=== RESEARCH RESULTS ===\n")
        print(response)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
