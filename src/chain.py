"""
chain.py
--------
Phase 3 (Tasks 10-12) — Prompt template, RAG chain, and conversation memory.

Task 10 : Finance-specific prompt template with {context} and {question}.
Task 11 : RAG chain wiring retriever + prompt + LLM (added in next task).
Task 12 : Conversation memory for follow-up questions (added in next task).
"""

from langchain_core.prompts import ChatPromptTemplate

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

# ChatPromptTemplate expects a list of (role, content) tuples.
# "system" sets the assistant persona; "human" carries the employee's question.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


if __name__ == "__main__":
    # Preview the rendered prompt with dummy values to verify placeholders.
    sample = prompt_template.invoke(
        {
            "context": "Segment: Government. Country: Germany. Profit: $16,185.00. Month: January 2014.",
            "question": "What was the profit for the Government segment in Germany?",
        }
    )
    for msg in sample.messages:
        print(f"[{msg.type.upper()}]\n{msg.content}\n")
