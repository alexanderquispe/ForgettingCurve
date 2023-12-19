import polars as pl
import pandas as pd

def process_data(file_path, num_obs_per_user):
    # Scanning and reading the CSV file
    result_df = pl.read_csv(file_path, ignore_errors=True)

    # Filter for English language learning
    result_df_en = result_df.filter(pl.col("learning_language") == "en")

    # Calculate the counts of each user_id in the English subset
    user_counts = result_df_en.groupby("user_id").agg(
        appearance_count=pl.count()
    )

    # Filter to keep only user_ids with appearance_count >= num_obs_per_user
    user_counts_filtered = user_counts.filter(pl.col("appearance_count") >= num_obs_per_user)

    # Join the filtered counts back to result_df_en
    result_df_en_filtered = result_df_en.join(user_counts_filtered, on="user_id")

    # Filter result_df_en to keep only rows with a non-null appearance_count
    result_df_en_filtered = result_df_en_filtered.filter(pl.col("appearance_count").is_not_null())

    # Define a custom function to slice the first N rows
    def slice_first_n(df):
        return df.sort("timestamp").head(num_obs_per_user)

    # Group by user_id, apply the custom function to each group
    result_df_en_topN = (
        result_df_en_filtered
        .groupby("user_id", maintain_order=True)
        .apply(slice_first_n)
    )

    # Convert to Pandas DataFrame
    result_df_en_topN_pd = result_df_en_topN.to_pandas()

    return result_df_en_topN_pd

# # Example usage
# file_path = "C:/Users/Alexander/Dropbox/halflife_regression_rl/0_data/learning_traces.13m.csv"
# result_df_en_top20 = process_data(file_path, 100)
# result_df_en_top20