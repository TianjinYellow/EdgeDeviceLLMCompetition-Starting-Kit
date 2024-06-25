import pandas as pd


columns = ['CommonsenseQA', 'BIG-Bench-Hard', 'GSM8K', 'LongBench', 'HumanEval', 'CHID', 'TruthfulQA', 'Throughput', 'Memory-Usage']

# replace 0 with your evaluated results.
data = {col: [0] for col in columns}


df = pd.DataFrame(data)

csv_filename = 'Results.csv'
df.to_csv(csv_filename, index=False)

print(f"DataFrame saved to {csv_filename}")
