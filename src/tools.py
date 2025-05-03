import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from .research_tool import ResearchTool


class WebSearchTool(ResearchTool):
    """Tool for web search simulation."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information. Input should be a search query.",
        )

    def run(self, query: str) -> str:
        """Simulate web search results."""
        # In a real implementation, this would use a search API like Google or Bing
        # For demo purposes, we'll return mock results
        mock_results = [
            {
                "title": f"Research on {query}",
                "snippet": f"Recent findings about {query} show promising results in the field.",
                "url": f"https://example.com/research/{query.replace(' ', '-').lower()}",
            },
            {
                "title": f"Analysis of {query} trends",
                "snippet": f"Experts analyze the latest trends in {query} and their implications.",
                "url": f"https://example.com/analysis/{query.replace(' ', '-').lower()}",
            },
            {
                "title": f"{query} - Wikipedia",
                "snippet": f"Comprehensive information about {query} including history and applications.",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_').lower()}",
            },
        ]

        results_text = "Search results:\n"
        for i, result in enumerate(mock_results):
            results_text += f"{i + 1}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}\n\n"

        return results_text


class WebScraperTool(ResearchTool):
    """Tool for web scraping."""

    def __init__(self):
        super().__init__(
            name="web_scraper",
            description="Scrape content from a webpage. Input should be a URL.",
        )
        self.headers = {"User-Agent": "Research Agent/1.0"}

    def run(self, url: str) -> str:
        """Scrape content from the given URL."""
        try:
            # In production, respect robots.txt and rate limiting
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title = soup.title.text if soup.title else "No title found"

            # Extract main content (simplified)
            paragraphs = soup.find_all("p")
            content = "\n".join(
                [p.text for p in paragraphs[:5]]
            )  # Limit to first 5 paragraphs

            result = f"Title: {title}\n\nContent Summary:\n{content[:1000]}..."
            if len(content) > 1000:
                result += "\n[Content truncated for brevity]"

            return result

        except Exception as e:
            return f"Error scraping {url}: {str(e)}"


class DataAnalysisTool(ResearchTool):
    """Tool for analyzing numerical data."""

    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="Analyze numerical data. Input should be CSV data or a description of data to generate and analyze.",
        )

    def run(self, input_text: str) -> str:
        """Analyze data from the input."""
        try:
            # Check if input contains CSV data
            if "," in input_text and "\n" in input_text:
                # Parse CSV from input
                from io import StringIO

                df = pd.read_csv(StringIO(input_text))
            else:
                # Generate mock data based on description
                df = self._generate_mock_data(input_text)

            # Perform basic analysis
            analysis_results = self._analyze_dataframe(df)
            return analysis_results

        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    def _generate_mock_data(self, description: str) -> pd.DataFrame:
        """Generate mock data based on description."""
        # Simple mock data generation - in a real system, this would be more sophisticated
        if "time series" in description.lower():
            dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
            values = np.random.normal(100, 15, size=30).cumsum()
            return pd.DataFrame({"date": dates, "value": values})

        elif "correlation" in description.lower():
            x = np.random.normal(0, 1, size=100)
            # Create correlated y with some noise
            y = x * 0.8 + np.random.normal(0, 0.5, size=100)
            return pd.DataFrame({"x": x, "y": y})

        else:
            # Default dataset
            data = {
                "category": ["A", "B", "C", "D", "E"],
                "value1": np.random.randint(10, 100, size=5),
                "value2": np.random.randint(20, 200, size=5),
            }
            return pd.DataFrame(data)

    def _analyze_dataframe(self, df: pd.DataFrame) -> str:
        """Perform analysis on the dataframe."""
        result = "Data Analysis Results:\n\n"

        # Basic info
        result += f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Summary statistics
        result += "Summary Statistics:\n"
        result += df.describe().to_string() + "\n\n"

        # Check for correlations if numerical columns > 1
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            result += "Correlation Matrix:\n"
            result += df[num_cols].corr().to_string() + "\n\n"

        # Simple visualization description
        if len(num_cols) > 0:
            result += "Visualization Insights:\n"
            for col in num_cols[:3]:  # Limit to first 3 numerical columns
                mean_val = df[col].mean()
                std_val = df[col].std()
                result += f"- {col}: Mean={mean_val:.2f}, StdDev={std_val:.2f}\n"

                # Detect outliers
                outliers = df[abs(df[col] - mean_val) > 2 * std_val]
                if not outliers.empty:
                    result += f"  Found {len(outliers)} potential outliers in {col}.\n"

        return result


class ContentGeneratorTool(ResearchTool):
    """Tool for generating research summaries and content."""

    def __init__(self, llm):
        super().__init__(
            name="content_generator",
            description="Generate research summaries or content based on information. Input should be a description of what content to generate.",
        )
        self.llm = llm

        # Template for content generation
        self.template = """
        Generate {content_type} based on the following information:

        INFORMATION:
        {information}

        INSTRUCTIONS:
        {instructions}

        Generate high-quality, informative content based on the above.
        """
        self.prompt = PromptTemplate(
            input_variables=["content_type", "information", "instructions"],
            template=self.template,
        )

    def run(self, input_text: str) -> str:
        """Generate content based on input description."""
        try:
            # Parse the input to extract the content type and information
            lines = input_text.strip().split("\n")

            content_type = "research summary"
            information = input_text
            instructions = "Create a concise research summary"

            # Try to extract structured information if available
            for line in lines:
                if line.lower().startswith("type:"):
                    content_type = line[5:].strip()
                elif line.lower().startswith("info:"):
                    information = line[5:].strip()
                elif line.lower().startswith("instructions:"):
                    instructions = line[13:].strip()

            # Generate content
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            content = chain.run(
                content_type=content_type,
                information=information,
                instructions=instructions,
            )

            return f"Generated {content_type}:\n\n{content}"

        except Exception as e:
            return f"Error generating content: {str(e)}"
