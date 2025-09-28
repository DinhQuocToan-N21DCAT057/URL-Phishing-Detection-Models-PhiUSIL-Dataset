# import pandas as pd
# import os

# from url_features_extractor_static import URL_EXTRACTOR

# url = "https://uis.ptithcm.edu.vn/#/home"

# temp = []
# extractor = URL_EXTRACTOR(url)
# data = extractor.extract_to_dataset()
# print(extractor.exec_time)
# temp.append(data)

# test = pd.DataFrame(temp)
# test.to_csv("test.csv", index=False)

import os
import pandas as pd
import csv

BASE_DIR = "D:/Hoc Tap/Giao Trinh va Bai Tap/2024-2025/DeAnTotNghiep/URL-Phishing-Detection-Models-PhiUSIL-Dataset"

def merge_files(df, second_file, output_file):
    """Merge two CSV files based on 'url' column, ensuring consistent labels and combining features"""
    try:
        # Read the second CSV file with semicolon delimiter
        second_df = pd.read_csv(
            second_file, 
            sep=",", 
            engine="python", 
            encoding="utf-8-sig", 
            quoting=csv.QUOTE_ALL, 
            on_bad_lines='skip'
        )

        # Verify required columns
        required_cols = ["url", "label"]
        for df_name, df_check in [("first file", df), ("second file", second_df)]:
            missing_cols = [col for col in required_cols if col not in df_check.columns]
            if missing_cols:
                print(f"Error: Columns {missing_cols} not found in {df_name}.")
                return df

        # Ensure URLs are strings and stripped
        df["url"] = df["url"].astype(str).str.strip()
        second_df["url"] = second_df["url"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        second_df["label"] = second_df["label"].astype(int)

        # Check for label consistency
        merged_check = pd.merge(
            df[["url", "label"]],
            second_df[["url", "label"]],
            on="url",
            how="inner",
            suffixes=("_first", "_second"),
        )
        label_mismatches = merged_check[merged_check["label_first"] != merged_check["label_second"]]
        if not label_mismatches.empty:
            print(f"Warning: Found {len(label_mismatches)} URLs with mismatched labels:")
            print(label_mismatches[["url", "label_first", "label_second"]])

        # Perform full merge (inner) to keep only URLs present in both files
        merged_df = pd.merge(
            df, second_df, on="url", how="inner", suffixes=("_first", "_second")
        )

        # If labels are identical, keep only one label column
        if "label_first" in merged_df.columns and "label_second" in merged_df.columns:
            if (merged_df["label_first"] == merged_df["label_second"]).all():
                merged_df = merged_df.drop(columns=["label_second"])
                merged_df = merged_df.rename(columns={"label_first": "label"})
            else:
                print("Warning: Labels are not consistent across files. Keeping both label columns.")

        # Reorder columns to ensure label(s) are at the end
        cols = merged_df.columns.tolist()
        if "label" in cols:
            cols.remove("label")
            cols.append("label")
        elif "label_first" in cols and "label_second" in cols:
            cols.remove("label_first")
            cols.remove("label_second")
            cols.extend(["label_first", "label_second"])
        merged_df = merged_df[cols]

        # Report URLs not present in both files
        urls_first_only = df[~df["url"].isin(second_df["url"])][["url", "label"]]
        urls_second_only = second_df[~df["url"].isin(df["url"])][["url", "label"]]
        if not urls_first_only.empty:
            print(f"URLs in first file but not in second file ({len(urls_first_only)} URLs):")
            print(urls_first_only)
        if not urls_second_only.empty:
            print(f"URLs in second file but not in first file ({len(urls_second_only)} URLs):")
            print(urls_second_only)

        # Save merged DataFrame to output file
        if output_file:
            merged_df.to_csv(output_file, index=False, sep=",", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            print(f"Merged data saved to {output_file}")

        return merged_df
    except Exception as e:
        print(f"Error merging files with '{second_file}': {e}")
        return df

if __name__ == "__main__":
    # Hardcode file paths
    dir_path = os.path.join(BASE_DIR, "Dataset")
    first_file = "PhiUSIIL_59_features_dataset.csv"
    second_file = "PhiUSIIL_Phishing_URL_Dataset_cleaned.csv"
    output_file = "final_PhiUSIIL_dataset.csv"

    # Read the first file (semicolon-delimited)
    first_file_path = os.path.join(dir_path, first_file)
    try:
        df = pd.read_csv(
            first_file_path, 
            sep=",", 
            engine="python", 
            encoding="utf-8-sig", 
            quoting=csv.QUOTE_ALL, 
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"Error reading '{first_file_path}': {e}")
        exit(1)

    # Merge with the second file (semicolon-delimited)
    second_file_path = os.path.join(dir_path, second_file)
    output_file_path = os.path.join(dir_path, output_file)
    merged_df = merge_files(df, second_file_path, output_file_path)
