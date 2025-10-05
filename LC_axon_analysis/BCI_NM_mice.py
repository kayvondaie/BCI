import pandas as pd

sheet_id = "15rnIwVV0hdLzp5gz0wOp2912v0HESNtZra2n8Cg2bys"
gid = "792846944"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# Read everything, including blank lines; keep as strings so we can trim
df = pd.read_csv(csv_url, engine="python", skip_blank_lines=False, dtype=str)

# Normalize column names just in case
df.columns = df.columns.str.strip()

# Keep only the columns we care about (if they exist)
needed = ["Name", "Genotype"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Strip whitespace and handle empties
df["Name"] = df["Name"].astype(str).str.strip()
df["Genotype"] = df["Genotype"].astype(str).str.strip()

# Drop rows where Name is blank/NaN
mask = df["Name"].notna() & (df["Name"] != "") & (df["Name"].str.lower() != "nan")
name_geno = df.loc[mask, ["Name", "Genotype"]].reset_index(drop=True)

print(name_geno)
