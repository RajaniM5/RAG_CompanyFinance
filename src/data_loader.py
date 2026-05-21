"""
data_loader.py
--------------
Phase 1 — Load, clean, and convert the company finance CSV into
LangChain Documents ready for embedding.

Responsibilities:
  - load_and_clean : reads Financials.csv, fixes column/value whitespace,
                     parses money strings to float, parses dates.
  - make_documents : converts each cleaned row into a LangChain Document
                     with prose content (for semantic search) and structured
                     metadata (for filtered queries).
"""

import pandas as pd
from langchain_core.documents import Document

# Path is relative to the project root (RAG_CompanyFinance/).
RAW_PATH = "dataset/Financials.csv"


def load_and_clean() -> pd.DataFrame:
    """Return a cleaned DataFrame from Financials.csv."""
    df = pd.read_csv(RAW_PATH)

    # The CSV has leading/trailing spaces in both column names and values.
    df.columns = df.columns.str.strip()
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # These columns arrive as strings like "$32,370.00" or "(1,000)" for negatives.
    # We strip currency symbols, commas, and convert parentheses to minus signs.
    money_cols = ["Units Sold", "Gross Sales", "Discounts", "Sales", "COGS", "Profit"]
    for col in money_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[$,\s]", "", regex=True)
            .str.replace(r"\((.+)\)", r"-\1", regex=True)  # accounting negatives: (1000) → -1000
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

    # Dates in the CSV follow "dd/mm/yy" — specifying format avoids a pandas
    # UserWarning and ensures consistent parsing across all rows.
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")

    return df


def make_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert each DataFrame row into a LangChain Document.

    page_content  — natural-language prose so the embedding model captures
                    financial context (segment, country, product, figures).
    metadata      — structured fields for ChromaDB filtered queries
                    (e.g. filter by country='Germany' or year=2014).
    """
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Segment: {row['Segment']}. "
            f"Country: {row['Country']}. "
            f"Product: {row['Product']}. "
            f"Discount Band: {row['Discount Band']}. "
            f"Units Sold: {row['Units Sold']:,.0f}. "
            f"Manufacturing Price: ${row['Manufacturing Price']:,}. "
            f"Sale Price: ${row['Sale Price']:,}. "
            f"Gross Sales: ${row['Gross Sales']:,.2f}. "
            f"Discounts: ${row['Discounts']:,.2f}. "
            f"Net Sales: ${row['Sales']:,.2f}. "
            f"COGS: ${row['COGS']:,.2f}. "
            f"Profit: ${row['Profit']:,.2f}. "
            f"Month: {row['Month Name']} {row['Year']}."
        )
        metadata = {
            "segment": row["Segment"],
            "country": row["Country"],
            "product": row["Product"],
            "year":    int(row["Year"]),
            "month":   int(row["Month Number"]),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


if __name__ == "__main__":
    df = load_and_clean()
    print(df.dtypes)
    print(df.head(3))
