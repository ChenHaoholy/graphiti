import json
import logging
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import DEFAULT_MODEL, OpenAIGenericClient
from graphiti_core.prompts.models import Message
import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

try:
    from .config import (
        get_embedding_api_key,
        get_embedding_base_url,
        get_embedding_dim,
        get_embedding_dimensions,
        get_embedding_model,
        get_llm_api_key,
        get_llm_base_url,
        get_llm_model,
        get_neo4j_password,
        get_neo4j_uri,
        get_neo4j_user,
        get_reranker_api_key,
        get_reranker_base_url,
        get_reranker_model,
        suppress_neo4j_notifications,
    )
except ImportError:
    from config import (
        get_embedding_api_key,
        get_embedding_base_url,
        get_embedding_dim,
        get_embedding_dimensions,
        get_embedding_model,
        get_llm_api_key,
        get_llm_base_url,
        get_llm_model,
        get_neo4j_password,
        get_neo4j_uri,
        get_neo4j_user,
        get_reranker_api_key,
        get_reranker_base_url,
        get_reranker_model,
        suppress_neo4j_notifications,
    )


class OpenAICompatibleEmbedder(OpenAIEmbedder):
    def __init__(self, config: OpenAIEmbedderConfig, dimensions: int | None = None):
        super().__init__(config=config)
        self.dimensions = dimensions

    async def create(self, input_data) -> list[float]:
        kwargs = {
            "input": input_data,
            "model": self.config.embedding_model,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        result = await self.client.embeddings.create(**kwargs)
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        kwargs = {
            "input": input_data_list,
            "model": self.config.embedding_model,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        result = await self.client.embeddings.create(**kwargs)
        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]


class JSONModeOpenAICompatibleClient(OpenAIGenericClient):
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        openai_messages: list[ChatCompletionMessageParam] = []

        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            openai_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Return only one valid json object. Do not wrap it in Markdown. "
                        f"The json object must match this JSON Schema: {schema}"
                    ),
                }
            )

        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role in {"system", "user"}:
                openai_messages.append({"role": message.role, "content": message.content})

        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.RateLimitError as exc:
            raise RateLimitError from exc

        result = response.choices[0].message.content or "{}"
        return json.loads(result)


def get_graphiti_client() -> Graphiti:
    if suppress_neo4j_notifications():
        logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    llm_config = LLMConfig(
        api_key=get_llm_api_key(),
        model=get_llm_model(),
        small_model=get_llm_model(),
        base_url=get_llm_base_url(),
        temperature=0,
    )
    llm_client = JSONModeOpenAICompatibleClient(config=llm_config)
    embedder = OpenAICompatibleEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=get_embedding_api_key(),
            base_url=get_embedding_base_url(),
            embedding_model=get_embedding_model(),
            embedding_dim=get_embedding_dim(),
        ),
        dimensions=get_embedding_dimensions(),
    )
    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            api_key=get_reranker_api_key(),
            model=get_reranker_model(),
            base_url=get_reranker_base_url(),
            temperature=0,
        )
    )

    return Graphiti(
        uri=get_neo4j_uri(),
        user=get_neo4j_user(),
        password=get_neo4j_password(),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
