"""Router Agent: Semantic intent classification using Google Gemini structured output."""

from typing import Literal, Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from config import config
from prompts import ROUTER_SYSTEM_PROMPT
from utils.logger import get_logger
from utils.exceptions import RouterError

logger = get_logger("router_agent", config.log_level)


class RouterDecision(BaseModel):
    """Pydantic Schema for structured LLM Router Agent classification."""

    route: Literal["RAG", "MCP"] = Field(
        ...,
        description="Target execution subsystem: 'RAG' for document questions, 'MCP' for tools (math, time, uuid, file reading)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    reasoning: str = Field(
        ...,
        description="Detailed semantic rationale for why this route was selected."
    )


class RouterAgent:
    """Semantic Intent Classification Agent powered by Google Gemini."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """Initializes Router Agent with ChatGoogleGenerativeAI.

        Args:
            model_name: Model string (defaults to config.llm_model_name).
            api_key: Gemini API key.
        """
        self.model_name = model_name or config.llm_model_name
        self.api_key = api_key or config.api_key

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
        )
        self.structured_llm = self.llm.with_structured_output(RouterDecision)

    def classify(self, user_query: str, history: Optional[List[BaseMessage]] = None) -> RouterDecision:
        """Classifies the semantic intent of a query synchronously.

        Args:
            user_query: Active user request.
            history: Optional conversation message history.

        Returns:
            RouterDecision instance with route, confidence, and reasoning.
        """
        logger.info(f"RouterAgent classifying user query: '{user_query}'")
        try:
            messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
            if history:
                messages.extend(history[-4:])  # Include recent context for multi-turn intent
            messages.append(HumanMessage(content=user_query))

            decision = self.structured_llm.invoke(messages)

            # Fallback check if response wasn't converted automatically
            if isinstance(decision, dict):
                decision = RouterDecision(**decision)

            logger.info(
                f"RouterAgent Decision: Route='{decision.route}', "
                f"Confidence={decision.confidence:.2f}, Reasoning='{decision.reasoning}'"
            )
            return decision

        except Exception as e:
            logger.error(f"RouterAgent LLM classification failed: {e}", exc_info=True)
            # Heuristic fallback if LLM is unavailable or rate-limited
            query_lower = user_query.lower()
            is_tool = any(kw in query_lower for kw in ["time", "clock", "date", "calculate", "math", "uuid", "read file"])
            fallback_route = "MCP" if is_tool else "RAG"
            
            logger.warning(f"Using RouterAgent fallback decision: {fallback_route}")
            return RouterDecision(
                route=fallback_route,
                confidence=0.5,
                reasoning=f"LLM Classification Fallback due to error: {str(e)}"
            )

    async def aclassify(self, user_query: str, history: Optional[List[BaseMessage]] = None) -> RouterDecision:
        """Classifies the semantic intent of a query asynchronously."""
        logger.info(f"RouterAgent async classifying user query: '{user_query}'")
        try:
            messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
            if history:
                messages.extend(history[-4:])
            messages.append(HumanMessage(content=user_query))

            decision = await self.structured_llm.ainvoke(messages)

            if isinstance(decision, dict):
                decision = RouterDecision(**decision)

            logger.info(
                f"RouterAgent Async Decision: Route='{decision.route}', "
                f"Confidence={decision.confidence:.2f}, Reasoning='{decision.reasoning}'"
            )
            return decision

        except Exception as e:
            logger.error(f"RouterAgent async classification failed: {e}", exc_info=True)
            query_lower = user_query.lower()
            is_tool = any(kw in query_lower for kw in ["time", "clock", "date", "calculate", "math", "uuid", "read file"])
            fallback_route = "MCP" if is_tool else "RAG"
            
            return RouterDecision(
                route=fallback_route,
                confidence=0.5,
                reasoning=f"LLM Async Classification Fallback due to error: {str(e)}"
            )
