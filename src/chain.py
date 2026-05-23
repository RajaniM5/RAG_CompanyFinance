"""
chain.py
--------
Phase 3 (Tasks 10-12) + Phase 4 — RAG chain with conversation memory.

Task 10 : Finance-specific prompt template with {context} and {question}.
Task 11 : RAG chain wiring retriever + prompt + Ollama LLM (llama3.2).
Task 12 : ChatMessageHistory for follow-up question support.
           ConversationBufferMemory is deprecated in LangChain 0.3+;
           ChatMessageHistory + RunnableWithMessageHistory is the current API.
Phase 4 : Test queries and retrieval evaluation via run_test_queries().
"""

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import OllamaLLM
from src.retriver import get_retriever, get_filtered_retriever

# ---------------------------------------------------------------------------
# Task 10 — Finance-specific prompt template
# ---------------------------------------------------------------------------

# The system prompt positions the LLM as a financial analyst so it answers
# in business terms (segments, margins, revenue) rather than generic text.
# Instructions keep answers grounded in the retrieved data only — this prevents
# the LLM from hallucinating figures that aren't in the dataset.
SYSTEM_PROMPT = """You are a financial analyst assistant for the company. \
Your job is to answer employee questions about company financial performance \
using only the data provided in the context below.

Guidelines:
- Base your answers strictly on the context provided. Do not make up numbers.
- When comparing segments, countries, or products, summarise the key figures clearly.
- If the context does not contain enough information to answer the question, \
say "I don't have enough data to answer that."
- Format currency values with $ and commas (e.g. $32,370.00).
- Keep answers concise and professional.

Context:
{context}"""

# MessagesPlaceholder injects the full chat history at the "history" slot so
# the LLM resolves follow-up questions (e.g. "What about France?") in context.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Task 11 — RAG chain (retriever + prompt + LLM)
# ---------------------------------------------------------------------------

# llama3.2 runs locally via Ollama — no API key required.
# temperature=0 keeps financial answers deterministic and factual.
llm = OllamaLLM(model="llama3.2", temperature=0)


def _format_docs(docs) -> str:
    """Join retrieved document contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_base_chain(segment: str | None = None,
                      country: str | None = None,
                      year: int | None = None):
    """Wire retriever + prompt + LLM into a runnable chain."""
    if any([segment, country, year]):
        retriever = get_filtered_retriever(segment=segment, country=country, year=year)
    else:
        retriever = get_retriever()

    # itemgetter extracts individual keys from the input dict.
    # RunnableWithMessageHistory passes {"question": ..., "history": [...]} to
    # this chain, so each key must be pulled out explicitly before the prompt.
    return (
        {
            "context":  itemgetter("question") | retriever | _format_docs,
            "question": itemgetter("question"),
            "history":  itemgetter("history"),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

# ---------------------------------------------------------------------------
# Task 12 — Conversation memory (ChatMessageHistory)
# ---------------------------------------------------------------------------

# One history store per session_id — supports multiple concurrent users.
# Using a plain dict here; swap for Redis/DB in production.
_session_histories: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the message history for a session."""
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


def build_chain(segment: str | None = None,
                country: str | None = None,
                year: int | None = None) -> RunnableWithMessageHistory:
    """
    Build and return a RAG chain with conversation memory and optional filters.

    The chain flow:
      question + history → retriever → prompt → LLM → answer
                                ↑
                        stored in ChatMessageHistory

    Args:
        segment : restrict retrieval to a specific segment (e.g. "Government").
        country : restrict retrieval to a specific country (e.g. "Germany").
        year    : restrict retrieval to a specific year (e.g. 2014).
    """
    base_chain = _build_base_chain(segment=segment, country=country, year=year)

    return RunnableWithMessageHistory(
        base_chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def ask(question: str,
        session_id: str = "default",
        segment: str | None = None,
        country: str | None = None,
        year: int | None = None) -> str:
    """
    Ask a question and get an answer grounded in the finance dataset.

    Conversation history is tracked per session_id so follow-up questions
    like "What about France?" resolve correctly.

    Args:
        question   : employee's natural-language question.
        session_id : identifier for the conversation session.
        segment    : optional metadata filter.
        country    : optional metadata filter.
        year       : optional metadata filter.
    """
    chain = build_chain(segment=segment, country=country, year=year)
    return chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )


def reset_session(session_id: str = "default"):
    """Clear conversation history for a session."""
    if session_id in _session_histories:
        _session_histories[session_id].clear()

# ---------------------------------------------------------------------------
# Phase 4 — Test queries
# ---------------------------------------------------------------------------

# Covers unfiltered queries, segment comparison, country aggregation,
# and a follow-up question to validate conversation memory.
TEST_QUERIES = [
    "Which country had the highest profit overall?",
    "Compare Government vs Midmarket segment sales.",
    "What was the total revenue for Germany?",
    "Which product generated the most profit?",
    "What about France?",   # follow-up — relies on memory of previous question
]


def run_test_queries():
    """Run Phase 4 test queries and print answers to evaluate retrieval quality."""
    print("=" * 60)
    print("Phase 4 — RAG Test Queries")
    print("=" * 60)
    reset_session("test")
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQ{i}: {query}")
        print("-" * 40)
        answer = ask(query, session_id="test")
        print(f"A:  {answer}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_test_queries()
