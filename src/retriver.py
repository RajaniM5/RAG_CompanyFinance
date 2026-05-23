"""
retriver.py
-----------
Phase 3 (Task 9) — Wrap the ChromaDB vector store as a LangChain retriever.

Provides:
  - get_retriever      : basic retriever returning top-k results.
  - get_filtered_retriever : retriever scoped to a specific segment,
                             country, or year via ChromaDB metadata filters.
"""

from langchain_core.vectorstores import VectorStoreRetriever
from src.embedder import load_vectorstore

# Number of documents to retrieve per query.
# 5 gives the LLM enough context without exceeding token limits.
DEFAULT_K = 5


def get_retriever(k: int = DEFAULT_K) -> VectorStoreRetriever:
    """
    Return a retriever over all 700 finance documents.

    Args:
        k: number of top results to return per query.
    """
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_filtered_retriever(
    segment: str | None = None,
    country: str | None = None,
    year: int | None = None,
    k: int = DEFAULT_K,
) -> VectorStoreRetriever:
    """
    Return a retriever scoped to a subset of documents using metadata filters.

    ChromaDB applies the filter before similarity search, so all k results
    will match the specified criteria. Only non-None arguments are included
    in the filter — passing no arguments is equivalent to get_retriever().

    Args:
        segment : e.g. "Government", "Midmarket", "Enterprise"
        country : e.g. "Germany", "France", "Canada"
        year    : e.g. 2014
        k       : number of top results to return per query.

    Example:
        retriever = get_filtered_retriever(country="Germany", year=2014)
    """
    # Build individual filter clauses only for provided arguments.
    clauses = []
    if segment:
        clauses.append({"segment": segment})
    if country:
        clauses.append({"country": country})
    if year:
        clauses.append({"year": year})

    # ChromaDB requires $and when combining multiple conditions;
    # a single clause can be passed directly without wrapping.
    if len(clauses) == 0:
        where_filter = {}
    elif len(clauses) == 1:
        where_filter = clauses[0]
    else:
        where_filter = {"$and": clauses}

    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            **({"filter": where_filter} if where_filter else {}),
        },
    )


if __name__ == "__main__":
    # Smoke test — unfiltered retrieval
    retriever = get_retriever()
    results = retriever.invoke("Which product had the highest profit?")
    print(f"Unfiltered — {len(results)} results")
    for doc in results:
        print(" ", doc.page_content[:120])

    print()

    # Smoke test — filtered by country and year
    retriever = get_filtered_retriever(country="Germany", year=2014)
    results = retriever.invoke("total sales")
    print(f"Filtered (Germany, 2014) — {len(results)} results")
    for doc in results:
        print(" ", doc.metadata, "|", doc.page_content[:80])
