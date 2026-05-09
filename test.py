import pandas as pd
df = pd.read_parquet("unscaled_test.parquet")
# one row per attack class
sample_rows = df.groupby("Attack").head(1)

print(sample_rows)