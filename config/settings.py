from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_model: str = "gpt-4o"
    secondary_model: str = "claude-sonnet-4-5"

    # MCP Server URLs
    web_search_mcp_url: str = "http://localhost:8001/sse"
    arxiv_mcp_url: str = "http://localhost:8002/sse"
    github_mcp_url: str = "http://localhost:8003/sse"
    mcp_jwt_secret: str = ""

    # External APIs
    tavily_api_key: str = ""
    github_token: str = ""

    # Tracing
    langchain_api_key: str | None = None
    langchain_tracing_v2: bool = True
    langchain_project: str = "deep-research-agent"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Agent limits
    max_iterations: int = 15
    max_cost_per_run_usd: float = 2.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
