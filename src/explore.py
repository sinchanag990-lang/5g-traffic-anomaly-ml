import pandas as pd

df = pd.read_csv("data/Combined.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))
print(df.iloc[:, -1].value_counts(dropna=False))