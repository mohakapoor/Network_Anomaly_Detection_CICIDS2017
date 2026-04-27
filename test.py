import polars as pl

df = pl.read_parquet(r'unscaled_test.parquet')
print(df["Attack"].value_counts())