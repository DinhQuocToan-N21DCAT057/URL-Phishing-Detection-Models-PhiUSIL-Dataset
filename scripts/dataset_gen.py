import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import os
import argparse
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Generator')
    parser.add_argument('--file', type=str, required=True, help='Dataset file')
    parser.add_argument('--start_idx', type=int, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, help='End index (exclusive)')
    parser.add_argument('--checkpoint_step', type=int, help='Checkpoint per step', default=200)
    args = parser.parse_args()

    if args.file:
        DATASET_PATH = os.path.join(BASE_DIR, args.file)
    else:
        print(f"File {args.file} not found!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, sep=";", engine="python", quotechar='"', encoding="utf-8")
    total = len(df)
    temp = []

    if args.start_idx is not None and args.end_idx is not None:
        # Slice the DataFrame according to the provided indices
        sliced_df = df.iloc[args.start_idx:args.end_idx]
        print(f"Extracting urls from {args.start_idx} to {args.end_idx} ({len(sliced_df)})")
        
        temp = []
        last_checkpoint = args.start_idx

        for batch_idx, item in enumerate(sliced_df.itertuples(index=True), 1):
            print("="*150)
            print(f"[{batch_idx}/{len(sliced_df)}] (global idx: {item.Index}) Extracting features for:")
            print(f"  URL  : {item.url}")
            print(f"  Label: {item.label}")
            try:
                extractor = URL_EXTRACTOR(item.url, item.label)
                data = extractor.extract_to_dataset()

                # nếu crawler fail (ví dụ content_features is_alive = 0) thì bỏ qua
                if extractor.content_features.get("is_alive", 1) == 0:
                    print(f"  ⚠️ Skipping URL '{item.url}' (site not alive or crawler error)")
                    continue

                print(f"  URL '{item.url}' took '{round(extractor.exec_time, 2)}' seconds to extract")
                temp.append(data)
            except Exception as e:
                print(f"  ❌ Error extracting {item.url}: {e}")
                continue

            # ⏩ Checkpoint flush (luôn ghi, kể cả khi temp rỗng)
            is_checkpoint = (batch_idx % args.checkpoint_step == 0)
            is_last = (batch_idx == len(sliced_df))

            if is_checkpoint or is_last:
                checkpoint_end = args.start_idx + batch_idx
                out_file = f"final_dataset_{args.start_idx}_{checkpoint_end}.csv"
                df_checkpoint = pd.DataFrame(temp)

                # nếu không có record nào thì vẫn tạo file rỗng để đánh dấu checkpoint
                df_checkpoint.to_csv(out_file, index=False)
                print(f"  💾 Saved checkpoint: {out_file} (rows={len(df_checkpoint)})")


    else:
        print(f"Extracting url all {len(df)} urls")
        for idx, item in enumerate(df.itertuples(index=False), 1):
            print("="*150)
            print(f"[{idx}/{total}] Extracting features for:")
            print(f"  URL  : {item.url}")
            print(f"  Label: {item.label}")
            try:
                extractor = URL_EXTRACTOR(item.url, item.label)
                data = extractor.extract_to_dataset()

                # nếu crawler fail (ví dụ content_features is_alive = 0) thì bỏ qua
                if extractor.content_features.get("is_alive", 1) == 0:
                    print(f"  ⚠️ Skipping URL '{item.url}' (site not alive or crawler error)")
                    continue

                print(f"  URL '{item.url}' took '{round(extractor.exec_time, 2)}' seconds to extract")
                temp.append(data)
            except Exception as e:
                print(f"  ❌ Error extracting {item.url}: {e}")
                continue

        print("="*150)
        final_dataset = pd.DataFrame(temp)
        final_dataset.to_csv("final_dataset.csv", index=False)