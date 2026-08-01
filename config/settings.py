from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM & API Keys
    openai_api_key: str = ""

    # Central Model Configuration (non-sensitive defaults defined in config)
    default_model: str = "gpt-5-mini"
    secondary_model: str = "gpt-5-mini"

    # Per-node temperature overrides
    classifier_temperature: float = 0.0  # deterministic classification
    planner_temperature: float = 0.2  # slight creativity in decomposition
    synthesis_temperature: float = 0.3  # creative synthesis
    critic_temperature: float = 0.0  # deterministic evaluation

    # MCP Server URLs
    web_search_mcp_url: str = "http://localhost:8001/sse"
    arxiv_mcp_url: str = "http://localhost:8002/sse"
    github_mcp_url: str = "http://localhost:8003/sse"
    mcp_jwt_secret: str = "deep-research-agent-mcp-jwt-secret-key-2026"

    # External APIs
    tavily_api_key: str = ""
    github_token: str = ""

    # Tracing
    langchain_api_key: str | None = None
    langchain_tracing_v2: bool = True
    langchain_project: str = "deep-research-agent"

    # Agent limits
    max_iterations: int = 10
    max_cost_per_run_usd: float = 0.5

    # Node LLM Timeouts (seconds)
    classifier_timeout_seconds: float = 60.0
    planner_timeout_seconds: float = 90.0
    critic_timeout_seconds: float = 60.0
    synthesis_timeout_seconds: float = 120.0
    evaluator_timeout_seconds: float = 90.0

    # Agent Concurrency & MCP Settings
    agent_max_concurrency: int = 10
    mcp_connect_timeout_seconds: float = 15.0

    # Cost Estimator Settings
    pricing_cache_max_age_seconds: int = 7 * 24 * 60 * 60
    pricing_fetch_timeout_seconds: float = 10.0

    # Frontend UI Settings
    avg_tokens_per_char: float = 0.25

    # API Server
    agent_api_url: str = "http://localhost:8080"
    agent_api_port: int = 8080
    frontend_password: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
