import pandas as pd
import os
import sys

# Set console encoding to UTF-8 to handle Vietnamese characters
if sys.platform.startswith("win"):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Define the base directory and dataset paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH1 = os.path.join(BASE_DIR, "Dataset", "top-1m_url_benign.csv")
DATASET_PATH2 = os.path.join(BASE_DIR, "Dataset", "balanced_dataset_1.csv")
DATASET_PATH3 = os.path.join(BASE_DIR, "Dataset", "dataset_phishing.csv")
DATASET_PATH4 = os.path.join(BASE_DIR, "Dataset", "phishing_urls_final.csv")

# Read the datasets
phishing_set = pd.read_csv(DATASET_PATH4, sep=";")
benign_set1 = pd.read_csv(DATASET_PATH1, sep=";")
df2 = pd.read_csv(DATASET_PATH2, sep=",")
df3 = pd.read_csv(DATASET_PATH3, sep=",")

# Remove duplicates within each dataset based on 'url'
benign_set1 = benign_set1.drop_duplicates(subset=["url"]).copy()
benign_set1.drop(columns=["rank"], inplace=True)  # Drop 'rank' after deduplication
df2 = df2.drop_duplicates(subset=["url"]).copy()
df3 = df3.drop_duplicates(subset=["url"]).copy()

# Extract benign URLs from df2 and df3, add label=1
benign_set2 = df2[df2["label"] == "benign"][["url"]].copy()
benign_set2["label"] = 1

benign_set3 = df3[df3["status"] == "legitimate"][
    ["url"]
].copy()  # Adjust if 'benign' is used
benign_set3["label"] = 1

# Remove duplicates within benign_set2 and benign_set3
benign_set2 = benign_set2.drop_duplicates(subset=["url"]).copy()
benign_set3 = benign_set3.drop_duplicates(subset=["url"]).copy()

# Total phishing samples
N = len(phishing_set)

# Get sizes after deduplication
s1 = len(benign_set1)
s2 = len(benign_set2)
s3 = len(benign_set3)

# Contribute all from benign_set3
contrib3 = min(s3, N)
benign3_sample = benign_set3.head(contrib3)  # Use head since size is small

# Remaining needed
remaining = N - contrib3

# Minimum from benign_set1 (70% of total N)
min_contrib1 = int(0.7 * N)

# Adjust contributions
if min_contrib1 > remaining:
    contrib1 = min(s1, remaining)
    contrib2 = 0
else:
    contrib1 = min_contrib1
    contrib2 = remaining - contrib1
    # Adjust if contrib2 > s2
    if contrib2 > s2:
        contrib2 = min(s2, remaining - min_contrib1)
        contrib1 = remaining - contrib2
    # Ensure contrib1 <= s1
    if contrib1 > s1:
        contrib1 = min(s1, remaining)
        contrib2 = remaining - contrib1

# Sample from benign_set1 (top-ranked) and benign_set2
benign1_sample = benign_set1.head(contrib1)  # Take top contrib1 URLs (sorted by rank)
benign2_sample = (
    benign_set2.sample(contrib2, random_state=42)
    if contrib2 < s2
    else benign_set2.head(contrib2)
)

# Combine all benign samples and remove duplicates
benign_final = pd.concat(
    [benign1_sample, benign2_sample, benign3_sample], ignore_index=True
)
benign_final = benign_final.drop_duplicates(subset=["url"]).copy()

# Adjust if benign_final is slightly short due to duplicates
if len(benign_final) < N:
    shortfall = N - len(benign_final)
    # Try to fill from benign_set2 first
    available_set2 = benign_set2[~benign_set2["url"].isin(benign_final["url"])]
    additional_set2 = (
        available_set2.sample(min(shortfall, len(available_set2)), random_state=42)
        if len(available_set2) > 0
        else pd.DataFrame(columns=["url", "label"])
    )
    shortfall = N - len(benign_final) - len(additional_set2)
    # Then fill from benign_set1
    available_set1 = benign_set1[
        ~benign_set1["url"].isin(benign_final["url"])
        & ~benign_set1["url"].isin(additional_set2["url"])
    ]
    additional_set1 = (
        available_set1.head(shortfall)
        if shortfall > 0
        else pd.DataFrame(columns=["url", "label"])
    )
    benign_final = pd.concat(
        [benign_final, additional_set2, additional_set1], ignore_index=True
    )
    benign_final = benign_final.head(N)  # Ensure exact N samples

# Ensure phishing_set has 'label' column
if "label" not in phishing_set.columns:
    phishing_set = phishing_set.copy()
    phishing_set["label"] = 0

# Combine benign and phishing into final dataset
df_final = pd.concat([benign_final, phishing_set], ignore_index=True)

# Remove duplicates in final dataset (in case of overlap between benign and phishing)
df_final = df_final.drop_duplicates(subset=["url"]).copy()

# Save the final dataset
output_path = os.path.join(BASE_DIR, "Dataset", "PhiUSIIL_dataset_plus.csv")
df_final.to_csv(output_path, index=False, sep=";")

# Print summary
print(f"Đã lưu bộ dữ liệu cân bằng vào {output_path}")
print(
    f"Tổng benign: {len(benign_final)} (Set1: {len(benign1_sample)}, Set2: {len(benign2_sample)}, Set3: {len(benign3_sample)})"
)
print(f"Tổng phishing: {len(phishing_set)}")
print(f"Tổng dataset sau khi loại bỏ trùng lặp: {len(df_final)}")
