import polars as pl
import argparse
from pathlib import Path
import sys

def get_parquet_info(path: Path):
    """Prints record count and schema for a parquet file or directory."""
    try:
        if path.is_file():
            # Use scan_parquet for metadata-only access if possible
            # collect().height gives the row count
            count = pl.scan_parquet(path).select(pl.len()).collect().item()
            print(f"File: {path.name}")
            print(f"  Records: {count:,}")
            
            # Optional: Print schema
            # df = pl.read_parquet_schema(path)
            # print(f"  Schema: {df}")
            
        elif path.is_dir():
            files = list(path.glob("*.parquet"))
            if not files:
                print(f"No parquet files found in {path}")
                return
            
            total_count = 0
            print(f"Directory: {path}")
            for f in sorted(files):
                count = pl.scan_parquet(f).select(pl.len()).collect().item()
                print(f"  {f.name}: {count:,}")
                total_count += count
            
            print("-" * 20)
            print(f"Total Records: {total_count:,}")
            print(f"Total Files: {len(files)}")
        else:
            print(f"Error: Path {path} does not exist.")
            
    except Exception as e:
        print(f"Error processing {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check record counts in Parquet files.")
    parser.add_argument("path", type=str, help="Path to a parquet file or directory.")
    
    args = parser.parse_args()
    get_parquet_info(Path(args.path))

if __name__ == "__main__":
    main()
