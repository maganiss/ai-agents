from unittest.mock import MagicMock, patch
import pytest

from ai_agents.tech_news_agent import (
    DuckDuckGoSearchTool,
    news_task,
    tech_reporter,
    local_llm,
)


# ---------------------------------------------------------------------------
# 1. Test Custom Search Tool Wrapper
# ---------------------------------------------------------------------------
def test_duckduckgo_search_tool_initialization():
    """Verify tool metadata satisfies BaseTool Pydantic requirements."""
    tool = DuckDuckGoSearchTool()
    assert tool.name == "DuckDuckGo Search Tool"
    assert "Searches the web" in tool.description


@patch("ai_agents.tech_news_agent.DuckDuckGoSearchRun")
def test_duckduckgo_search_tool_run(mock_ddg_class):
    """Verify _run invokes the underlying LangChain tool correctly."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = "Search Results: Snowflake AI updates"
    mock_ddg_class.return_value = mock_instance

    tool = DuckDuckGoSearchTool()
    result = tool._run("Snowflake AI")

    mock_instance.invoke.assert_called_once_with("Snowflake AI")
    assert result == "Search Results: Snowflake AI updates"


# ---------------------------------------------------------------------------
# 2. Test Agent Configuration
# ---------------------------------------------------------------------------
def test_tech_reporter_agent_config():
    """Verify agent parameters, safety limits, and delegation settings."""
    assert tech_reporter.role == "Senior Tech News Researcher"
    assert tech_reporter.max_iter == 5
    assert tech_reporter.allow_delegation is False
    assert len(tech_reporter.tools) == 2


# ---------------------------------------------------------------------------
# 3. Test Task Configuration
# ---------------------------------------------------------------------------
def test_news_task_config():
    """Verify task output target and associated agent."""
    assert news_task.output_file == "tech_news_summary.md"
    assert news_task.agent == tech_reporter


# ---------------------------------------------------------------------------
# 4. Test Full Crew Execution (Mocked End-to-End)
# ---------------------------------------------------------------------------
@patch("ai_agents.tech_news_agent.Crew.kickoff")
def test_crew_kickoff(mock_kickoff):
    """Test kicking off the crew passes formatted topics correctly."""
    mock_kickoff.return_value = "Mocked Markdown Summary"

    from ai_agents.tech_news_agent import news_crew

    topics = ["Snowflake", "AI in retail"]
    formatted_topics = ", ".join(topics)

    result = news_crew.kickoff(inputs={"topics": formatted_topics})

    mock_kickoff.assert_called_once_with(inputs={"topics": "Snowflake, AI in retail"})
    assert result == "Mocked Markdown Summary"


@pytest.mark.integration
def test_ollama_live_connection():
    """Integration test: Verifies local Ollama server is running and model responds.

    Requires: 'ollama serve' running on http://localhost:11434 with 'llama3.1' pulled.
    """
    # Simple direct completion call to the local model instance
    response = local_llm.call(
        messages=[{"role": "user", "content": "Respond with 'OK'"}]
    )

    assert response is not None
    assert len(response) > 0
