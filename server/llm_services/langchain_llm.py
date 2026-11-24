import os

import httpx

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic


class LLM:
    def __init__(
        self,
        temperature: int = 0,
        streaming: bool = False,
        callbacks: list = [],
        max_retries=3,
        max_tokens=None,
        timeout=10
    ):
        self.temperature = temperature if temperature != 0 else self._get_role_temperature()
        self.streaming = streaming
        self.callbacks = callbacks
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.llm = self._init_llm()

    def _init_llm(self):
        llm = None
        if os.environ["CURRENT_MODEL"] == "Azure":
            llm = self._init_azure()
        if os.environ["CURRENT_MODEL"] == "OpenAI":
            llm = self._init_openai()
        if os.environ["CURRENT_MODEL"] == "Anthropic":
            llm = self._init_anthropic()
        return llm

    def _init_azure(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_BASE"],
            api_key=os.environ["AZURE_KEY"],
            api_version=os.environ["AZURE_VERSION"],
            azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
            temperature=self.temperature,
            streaming=self.streaming,
            callbacks=self.callbacks,
            max_retries=self.max_retries,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _init_openai(self) -> ChatOpenAI:
        model = os.environ["OPENAI_MODEL"]
        options = {
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ["OPENAI_BASE"] + "/v1",
            "organization": os.environ.get("OPENAI_ORGANIZATION") or None,
            "model": model,
            "streaming": self.streaming,
            "callbacks": self.callbacks,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

        proxy = os.environ.get("OPENAI_API_PROXY", "").strip()
        if proxy:
            options["http_client"] = httpx.Client(proxy=proxy)
            options["http_async_client"] = httpx.AsyncClient(proxy=proxy)

        if model.startswith(("gpt-6.2-", "gpt-6.1-", "gpt-6-")):
            # GPT-6 models use the Responses API and do not accept temperature.
            options["use_responses_api"] = True
            options["model_kwargs"] = {
                "max_output_tokens": self.max_tokens,
                "reasoning": {"effort": "low"},
            }
        else:
            # Keep Chat Completions behavior for older models.
            options["temperature"] = self.temperature
            options["max_tokens"] = self.max_tokens

        return ChatOpenAI(**options)

    def _init_anthropic(self) -> ChatAnthropic:
        return ChatAnthropic(
            anthropic_api_key=os.environ["ANTHROPIC_KEY"],
            model=os.getenv("ANTHROPIC_MODEL", "claude-2"),
            temperature=self.temperature,
            streaming=self.streaming,
            callbacks=self.callbacks,
            max_retries=self.max_retries,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _get_role_temperature(self):
        if (os.environ.get("CURRENT_ROLE") == "examiner"):
            return 0
        if (os.environ.get("CURRENT_ROLE") == "teacher"):
            return 0.5
        if (os.environ.get("CURRENT_ROLE") == "interviewer"):
            return 0.9
