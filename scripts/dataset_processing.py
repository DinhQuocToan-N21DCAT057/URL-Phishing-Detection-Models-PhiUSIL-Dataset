import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sys
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def chk_dup(df):
    df["URL"] = df["URL"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    duplicates = df[df.duplicated(subset=['URL'])]
    print(f"Found duplicated rows by 'URL':\n{duplicates}")
    return df.drop_duplicates(subset=['URL'])

def chk_stat_col(df, col_name):
    """Check statistic of specific column: count all values and display ratio using matplotlib"""
    
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in dataframe.")
        return df
    
    value_counts = df[col_name].value_counts(dropna=False)
    ratios = value_counts / value_counts.sum()
    print(f"Value counts for column '{col_name}':\n{value_counts}")
    print(f"Ratios for column '{col_name}':\n{ratios}")
    
    plt.figure(figsize=(8, 6))
    ratios.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False)
    plt.title(f"Ratio of values in '{col_name}'")
    plt.ylabel('')
    plt.show()
    return df

def chk_NaN(df):
    """Check NaN"""
    nan_counts = df.isna().sum()
    print("NaN counts per column:")
    print(nan_counts)
    rows_with_nan = df[df.isna().any(axis=1)]
    print(f"Rows with any NaN values ({len(rows_with_nan)} rows):")
    print(rows_with_nan)
    return df.dropna()

def chk_null(df):
    """Check null"""
    null_counts = df.isnull().sum()
    print("Null counts per column:")
    print(null_counts)
    rows_with_null = df[df.isnull().any(axis=1)]
    print(f"Rows with any null values ({len(rows_with_null)} rows):")
    print(rows_with_null)
    return df.drop(columns=rows_with_null.columns)

def drop_col(df, col_name):
    """Drop specific columns"""
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in dataframe. Skipping drop.")
        return df
    print(f"Dropping column: {col_name}")
    return df.drop(columns=[col_name])

def drop_all_col(df, except_cols):
    """Drop all columns except the specified columns"""
    # Check if all columns in except_cols exist
    missing_cols = [col for col in except_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in dataframe. Proceeding with available columns.")
    
    # Keep only the columns that exist in both except_cols and df.columns
    valid_except_cols = [col for col in except_cols if col in df.columns]
    if not valid_except_cols:
        print("No valid columns to keep. Returning original dataframe.")
        return df
    
    print(f"Dropping all columns except: {valid_except_cols}")
    return df[valid_except_cols]

def add_col(df, col_name, col_val):
    """Add specific value for columns"""
    if col_name in df.columns:
        print(f"Column '{col_name}' already exists. Overwriting with value: {col_val}")
    else:
        print(f"Adding column: {col_name} with value: {col_val}")
    df[col_name] = col_val
    return df

def ensure_url_label(df):
    """
    Trả về dataframe chỉ còn đúng 2 cột: URL (string) và label (int 0/1),
    dù file gốc có bị lệch cột hay bị gộp.
    """
    # 1) Nếu đã có sẵn 2 cột
    cols_lower = [c.lower().strip() for c in df.columns]
    colmap = {c.lower().strip(): c for c in df.columns}
    if "url" in cols_lower and "label" in cols_lower:
        URL_col = colmap["url"]
        LABEL_col = colmap["label"]
        out = df[[URL_col, LABEL_col]].copy()
        out.columns = ["URL", "label"]
    else:
        # 2) Thử khôi phục khi bị gộp vào 1 cột (thường là cột chứa URL)
        #    - Tìm cột có vẻ là URL (có "http")
        candidate_cols = [c for c in df.columns if df[c].astype(str).str.contains(r"http", case=False, na=False).any()]
        if candidate_cols:
            c = candidate_cols[0]
            s = df[c].astype(str)

            # Tách theo dấu phẩy cuối (URL,0/1)
            tmp = s.str.rsplit(",", n=1, expand=True)
            if tmp.shape[1] == 2 and tmp[1].str.strip().str.match(r"^[01]$").fillna(False).all():
                out = pd.DataFrame({"URL": tmp[0].str.strip(), "label": tmp[1].astype(int)})
            else:
                # Tách theo khoảng trắng cuối (URL 0/1)
                tmp = s.str.rsplit(r"\s+", n=1, expand=True, regex=True)
                if tmp.shape[1] == 2 and tmp[1].str.strip().str.match(r"^[01]$").fillna(False).all():
                    out = pd.DataFrame({"URL": tmp[0].str.strip(), "label": tmp[1].astype(int)})
                else:
                    # Nếu vẫn không được, lấy cột cuối làm label, cột có http làm URL
                    last_col_series = df.iloc[:, -1].astype(str)
                    lab = last_col_series.str.extract(r"([01])$")[0]
                    out = pd.DataFrame({"URL": s, "label": lab})
        else:
            # 3) Không tìm được cột có http ⇒ lấy cột đầu là URL, cột cuối là label
            s = df.iloc[:, 0].astype(str)
            last_col_series = df.iloc[:, -1].astype(str)
            lab = last_col_series.str.extract(r"([01])$")[0]
            out = pd.DataFrame({"URL": s, "label": lab})

    # Làm sạch cuối cùng
    out["URL"] = out["URL"].astype(str).str.strip()
    # ép label về {0,1}
    out["label"] = (out["label"].astype(str).str.extract(r"([01])$")[0]).fillna("0").astype(int)
    out["label"] = out["label"].clip(0, 1)
    return out


MODES = {
    'chk_dup': chk_dup,
    'chk_stat_col': chk_stat_col,
    'chk_NaN': chk_NaN,
    'chk_null': chk_null,
    'drop_col': drop_col,
    'drop_all_col': drop_all_col,
    'add_col': add_col,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Processing')
    parser.add_argument('--dir', type=str, required=True, help='Directory name of dataset')
    parser.add_argument('--modes', type=str, required=True, nargs='+', help='Processing modes')
    parser.add_argument('--cols', type=str, required=False, nargs='+', help='Specific columns')
    parser.add_argument('--except_cols', type=str, required=False, nargs='+', help='Exceptional columns')
    parser.add_argument('--val', type=str, required=False, help='Specific value for colums(add_col only)')
    parser.add_argument('--o', type=str, required=False, help='Output file name')
    parser.add_argument('--files', type=str, required=True, nargs='+', help='Processing files')
    args = parser.parse_args()
    
    dir_path = os.path.join(BASE_DIR, args.dir)
    if not os.path.isdir(dir_path):
        print(f"Directory not found: '{dir_path}'")
        sys.exit(1)

    if not args.files:
        print(f"No CSV files choosen: '{dir_path}'")
        sys.exit(1)

    if not args.modes or not all(mode in MODES for mode in args.modes):
        print(f"Invalid mode arguments chosen. Available modes: {list(MODES.keys())}")
        sys.exit(1)

    # Remove unnecessary initialization
    # df = []
    for file in args.files:
        try:
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(file_path, sep=",", engine="python", encoding="utf-8")
            df = ensure_url_label(df)

            # # Nếu label không tồn tại đúng cột vì bị dính Title → lấy cột cuối cùng làm label
            # if "label" not in df.columns:
            #     df["label"] = df.iloc[:, -1]
            #     df["label"] = df["label"].astype(str).str.extract(r'(\d)$').astype(int)

            # if "URL" in df.columns:
            #     df = df[["URL", "label"]]

            for mode in args.modes:
                if mode == 'chk_stat_col' or mode == 'drop_col':
                    if args.cols:
                        for col_name in args.cols:
                            df = MODES[mode](df, col_name)
                    else:
                        col_name = df.columns[0]
                        df = MODES[mode](df, col_name)
                elif mode == 'drop_all_col':
                    if args.except_cols:
                        df = MODES[mode](df, args.except_cols)
                    else:
                        print("drop_all_col mode requires --except_cols argument.")
                        sys.exit(1)
                elif mode == 'add_col':
                    if not args.cols or not args.val:
                        print("add_col mode requires both --cols and --val arguments.")
                        sys.exit(1)
                    # Support multiple columns and values
                    if isinstance(args.val, list) and len(args.cols) == len(args.val):
                        for col_name, col_val in zip(args.cols, args.val):
                            df = MODES[mode](df, col_name, col_val)
                    elif isinstance(args.val, list) and len(args.val) == 1:
                        for col_name in args.cols:
                            df = MODES[mode](df, col_name, args.val[0])
                    elif not isinstance(args.val, list):
                        for col_name in args.cols:
                            df = MODES[mode](df, col_name, args.val)
                    else:
                        print("Number of columns and values for add_col do not match.")
                        sys.exit(1)
                else:
                    df = MODES[mode](df)
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")
            sys.exit(1)

    if args.o:
        df.to_csv(args.o, index=False, sep=";", encoding="utf-8-sig")
