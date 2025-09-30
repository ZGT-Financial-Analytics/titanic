"""
GPU-Ready Code Template
Use this pattern in your analysis scripts for automatic GPU/CPU switching
"""


def setup_dataframe_engine():
    """Auto-detect and setup the best available DataFrame engine."""
    try:
        import cudf as df_engine

        engine_type = "cuDF (GPU)"
        print("🚀 GPU acceleration active with cuDF")
        return df_engine, "cudf"
    except ImportError:
        try:
            import polars as df_engine

            engine_type = "Polars (Fast CPU)"
            print("⚡ Fast CPU processing with Polars")
            return df_engine, "polars"
        except ImportError:
            import pandas as df_engine

            engine_type = "Pandas (Standard CPU)"
            print("🐼 Standard processing with Pandas")
            return df_engine, "pandas"


def read_data(filepath, engine_type):
    """Universal data reader that works with any engine."""
    df_engine, engine = setup_dataframe_engine()

    if engine == "cudf":
        return df_engine.read_csv(filepath)
    elif engine == "polars":
        return df_engine.read_csv(filepath)
    else:  # pandas
        return df_engine.read_csv(filepath)


# Example usage:
if __name__ == "__main__":
    df_engine, engine = setup_dataframe_engine()

    # This line works the same regardless of engine!
    df = read_data("data/raw/train.csv", engine)
    print(f"Loaded {len(df)} rows using {engine}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
