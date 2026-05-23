"""
main.py
-------
Phase 5 (Task 15) — Interactive CLI chat interface for employees.

Run from the project root:
    python3 -m app.main

Built-in commands:
    /filter   — set or clear segment / country / year filters
    /reset    — clear conversation history for a fresh session
    /quit     — exit the chat
"""

import uuid
from src.chain import ask, reset_session
from src.embedder import load_vectorstore
from src.retriver import get_retriever, get_filtered_retriever

# Valid values for filter validation — derived from the dataset.
VALID_SEGMENTS = {"Government", "Midmarket", "Channel Partners", "Enterprise", "Small Business"}
VALID_COUNTRIES = {"Canada", "Germany", "France", "Mexico", "United States of America"}
VALID_YEARS = {2013, 2014}


def print_banner():
    print("=" * 60)
    print("  Company Finance Assistant")
    print("  Powered by RAG + Ollama (llama3.2)")
    print("=" * 60)
    print("  Commands: /filter  /reset  /quit")
    print("  Ask anything about company financial performance.")
    print("=" * 60)
    print()


def prompt_filters() -> tuple[str | None, str | None, int | None]:
    """
    Interactively ask the employee if they want to scope the search.
    Returns (segment, country, year) — any can be None for no filter.
    """
    print("\n-- Optional Filters (press Enter to skip) --")

    segment = input(f"  Segment {VALID_SEGMENTS}: ").strip() or None
    if segment and segment not in VALID_SEGMENTS:
        print(f"  Unknown segment '{segment}' — filter cleared.")
        segment = None

    country = input(f"  Country {VALID_COUNTRIES}: ").strip() or None
    if country and country not in VALID_COUNTRIES:
        print(f"  Unknown country '{country}' — filter cleared.")
        country = None

    year_input = input(f"  Year {VALID_YEARS}: ").strip()
    year = None
    if year_input:
        try:
            year = int(year_input)
            if year not in VALID_YEARS:
                print(f"  Year {year} not in dataset — filter cleared.")
                year = None
        except ValueError:
            print("  Invalid year — filter cleared.")

    return segment, country, year


def show_sources(question: str,
                 segment: str | None,
                 country: str | None,
                 year: int | None):
    """Print the top-3 source documents used to answer the question."""
    if any([segment, country, year]):
        retriever = get_filtered_retriever(segment=segment, country=country, year=year, k=3)
    else:
        retriever = get_retriever(k=3)

    docs = retriever.invoke(question)
    print("\n  -- Sources --")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"  [{i}] {meta.get('segment')} | {meta.get('country')} | "
              f"{meta.get('product')} | {meta.get('month')}/{meta.get('year')}")
    print()


def main():
    print_banner()

    # Each CLI session gets a unique ID so history is isolated per run.
    session_id = str(uuid.uuid4())
    segment, country, year = None, None, None

    # Show active filters at startup if any were set.
    active = [f for f in [
        f"segment={segment}" if segment else None,
        f"country={country}" if country else None,
        f"year={year}" if year else None,
    ] if f]
    if active:
        print(f"  Active filters: {', '.join(active)}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            reset_session(session_id)
            print("  Conversation history cleared.\n")
            continue

        if user_input.lower() == "/filter":
            segment, country, year = prompt_filters()
            active = [f for f in [
                f"segment={segment}" if segment else None,
                f"country={country}" if country else None,
                f"year={year}" if year else None,
            ] if f]
            print(f"  Filters set: {', '.join(active) if active else 'none'}\n")
            continue

        # Get answer from RAG chain
        print("\nAssistant: ", end="", flush=True)
        answer = ask(
            question=user_input,
            session_id=session_id,
            segment=segment,
            country=country,
            year=year,
        )
        print(answer)

        # Show source documents used to ground the answer
        show_sources(user_input, segment, country, year)


if __name__ == "__main__":
    main()
