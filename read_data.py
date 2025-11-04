import pyarrow.parquet as pq

infos_type = pq.read_table('data/yellow_tripdata_2025-01.parquet')
df = infos_type.to_pandas()

print(infos_type.schema)
print(df.head())