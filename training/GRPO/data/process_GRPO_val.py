import pandas as pd

df = pd.read_parquet("parquets/train.parquet")

df_head = df.head(10000)

df_head.to_parquet("parquets/test.parquet", index=False)

print(f"Already extracted {len(df_head)}lines, saved to output_top10000.parquet")