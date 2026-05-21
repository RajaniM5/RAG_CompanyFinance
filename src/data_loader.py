import pandas as pd

RAW_PATH = "dataset/Financials.csv"

def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string values
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Numeric columns stored as strings — remove $ , and convert
    money_cols = ["Units Sold", "Gross Sales", "Discounts", "Sales", "COGS", "Profit"]
    for col in money_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[$,\s]", "", regex=True)
            .str.replace(r"\((.+)\)", r"-\1", regex=True)  # (1000) → -1000
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    return df

from langchain_core.documents import Document

def make_documents(df: pd.DataFrame) -> list[Document]:
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
            "segment":  row["Segment"],
            "country":  row["Country"],
            "product":  row["Product"],
            "year":     int(row["Year"]),
            "month":    int(row["Month Number"]),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


if __name__ == "__main__":
    df = load_and_clean()
    print(df.dtypes)
    print(df.head(3))