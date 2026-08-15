import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools import DuckDuckGoSearchRun


# ---------------------------------------------------------------------------
# 1. Custom CrewAI-Compatible DuckDuckGo Search Tool
# ---------------------------------------------------------------------------
# This wrapper inherits from CrewAI's BaseTool to satisfy Pydantic validation.
class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = (
        "Searches the web for a given query to find recent news, links, and articles."
    )

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)


# Instantiate the tools
search_tool = DuckDuckGoSearchTool()
scrape_tool = ScrapeWebsiteTool()


# ---------------------------------------------------------------------------
# 2. Local Model Configuration (Ollama)
# ---------------------------------------------------------------------------
# Low temperature forces factual accuracy and reduces hallucinated URLs.
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.1
)


# ---------------------------------------------------------------------------
# 3. Agent Definition
# ---------------------------------------------------------------------------
tech_reporter = Agent(
    role="Senior Tech News Researcher",
    goal="Search the web for the latest developments on {topics}, inspect web pages, and summarize key insights.",
    backstory=(
        "You are an expert technology journalist specializing in emerging tech trends. "
        "Your mission is to search the web for fresh news, read through high-value web pages, "
        "and present key findings in clear markdown format with direct links to sources."
    ),
    tools=[search_tool, scrape_tool],
    llm=local_llm,
    max_iter=5,             # Safety cap: stops loops if local model gets confused
    allow_delegation=False, # Essential stability setting for local LLMs
    verbose=True            # Displays real-time thought process in the terminal
)


# ---------------------------------------------------------------------------
# 4. Task Definition
# ---------------------------------------------------------------------------
news_task = Task(
    description=(
        "Search the internet for the latest news, updates, and major announcements regarding "
        "the following topics: {topics}.\n\n"
        "Instructions:\n"
        "1. Use the search tool to find 2-3 recent articles for each topic.\n"
        "2. Synthesize your findings into a clean, engaging tech roundup.\n"
        "3. For each topic, include:\n"
        "   - A clear subheader (`### Topic Name`)\n"
        "   - 3-4 concise bullet points detailing key news and developments.\n"
        "   - Markdown links back to the source websites where possible.\n"
        "   - Relevant Markdown image tags (`![Alt](URL)`) if article image URLs are discovered."
    ),
    expected_output=(
        "A structured Markdown report containing bulleted news updates grouped by topic, "
        "complete with source links and inline image markdown tags."
    ),
    agent=tech_reporter,
    output_file="tech_news_summary.md"  # Automatically saves to your hard drive
)


# ---------------------------------------------------------------------------
# 5. Assemble and Run
# ---------------------------------------------------------------------------
news_crew = Crew(
    agents=[tech_reporter],
    tasks=[news_task]
)

if __name__ == "__main__":
    print("🌐 Starting Local Tech News Agent...")

    # -----------------------------------------------------------------------
    # DEFINE YOUR TOPICS HERE:
    # Add or change any tech topics you want the agent to investigate.
    # -----------------------------------------------------------------------
    my_topics = [
        "Snowflake",
        "future of Data Engineering",
        "AI in the retail sector",
        "AI in enterprices"
    ]

    # Convert the Python list into a single comma-separated string
    formatted_topics = ", ".join(my_topics)

    # Kickoff execution and pass the string to the {topics} variable
    result = news_crew.kickoff(inputs={"topics": formatted_topics})

    print("\n✅ Task Complete!")
    print("Your report has been written to: tech_news_summary.md")