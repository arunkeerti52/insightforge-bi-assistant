"""InsightForge assistant: prompt chain (rephrase -> retrieve -> answer) with conversation memory."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*InMemoryChatMessageHistory.*")

from .knowledge_base import build_documents, build_vectorstore
from .llm import get_llm
from .retriever import BusinessStatsRetriever, detect_topics

REPHRASE_PROMPT = PromptTemplate.from_template(
    """You rewrite follow-up questions for a business-intelligence search system.
If the FOLLOW-UP QUESTION is already self-contained, return it EXACTLY unchanged.
Otherwise rewrite it as a self-contained question by resolving references ("it", "that region", "what about X")
using the PREVIOUS QUESTIONS. Never add numbers, facts or extra conditions. Return ONLY the question.

Examples:
Previous: "What were the total sales for Widget A?" | Follow-up: "What about Widget B?" -> "What were the total sales for Widget B?"
Previous: "Which product has the highest total sales?" | Follow-up: "Which region has the lowest sales?" -> "Which region has the lowest sales?"
Previous: "Show the yearly sales trend" | Follow-up: "And for the North region?" -> "Show the yearly sales trend for the North region"

PREVIOUS QUESTIONS:
{history}
FOLLOW-UP QUESTION: {question}
STANDALONE QUESTION:"""
)

ANSWER_PROMPT = PromptTemplate.from_template(
    """\
You are InsightForge, a senior business-intelligence analyst for a small/medium enterprise.
Rules:
1. Ground every number in the CONTEXT below - never invent figures. If the context is insufficient, say so.
2. Quote exact figures (currency with $ and thousands separators, percentages with 1 decimal).
3. Structure the reply as: **Answer:** (1-2 sentences), **Supporting statistics:** (bullets), **Recommendation:** (one actionable business suggestion).
   Put each of the three headings at the start of its own line.
4. Use the conversation history only to keep continuity, not as a data source.

CONVERSATION HISTORY:
{history}

CONTEXT START
{context}
CONTEXT END

QUESTION: {question}
ANSWER:"""
)


@dataclass
class Turn:
    question: str
    standalone: str
    answer: str
    sources: list
    topics: list
    latency_s: float


@dataclass
class InsightForgeAssistant:
    df: pd.DataFrame
    memory_window: int = 6
    k: int = 6
    turns: list = field(default_factory=list)

    def __post_init__(self):
        self.llm = get_llm()
        self.docs = build_documents(self.df)
        self.vectorstore = build_vectorstore(self.docs)
        self.retriever = BusinessStatsRetriever(vectorstore=self.vectorstore, df=self.df, k=self.k)
        self.memory = InMemoryChatMessageHistory()
        self.rephrase_chain = REPHRASE_PROMPT | self.llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    # -- memory ------------------------------------------------------------
    def history_text(self) -> str:
        msgs = self.memory.messages[-2 * self.memory_window:]
        return "\n".join(f"{'Human' if m.type == 'human' else 'AI'}: {m.content}" for m in msgs) or "(no previous conversation)"

    def previous_questions(self) -> str:
        qs = [m.content for m in self.memory.messages if m.type == "human"][-self.memory_window:]
        return "\n".join(f"- {q}" for q in qs) or "(none)"

    def reset_memory(self):
        self.memory.clear()
        self.turns.clear()

    # -- main entry ----------------------------------------------------------
    def ask(self, question: str, use_memory: bool = True) -> Turn:
        t0 = time.time()
        history = self.history_text() if use_memory else "(no previous conversation)"
        standalone = question
        if use_memory and self.memory.messages:
            standalone = self.rephrase_chain.invoke({"history": self.previous_questions(), "question": question}).strip().strip('"') or question
        docs = self.retriever.invoke(standalone)
        context = "\n".join(f"- {d.page_content}" for d in docs)
        answer = self.answer_chain.invoke({"history": history, "context": context, "question": standalone})
        if use_memory:
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)
        turn = Turn(question, standalone, answer, docs, detect_topics(standalone), round(time.time() - t0, 3))
        self.turns.append(turn)
        return turn
