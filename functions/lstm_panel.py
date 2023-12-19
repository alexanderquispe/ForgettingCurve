import os

# Define the path to the desired directory
directory_path = r'C:\Users\Alexander\Documents\GitHub\ForgettingCurve\functions'

# Change the current working directory to the specified path
os.chdir(directory_path)

# You can print the current working directory to confirm the change
print("Current Working Directory: ", os.getcwd())


# run_script.py
def main():
    with open('clean_data_en_lstm.py', 'r') as file:
        script = file.read()
        exec(script)

if __name__ == "__main__":
    main()

file_path = "C:/Users/Alexander/Dropbox/halflife_regression_rl/0_data/learning_traces.13m.csv"
result_df_en_top20 = process_data(file_path, 20)

from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import dask.dataframe as dd


def read_data(df, omit_lexemes=False):
    min_hl = 15.0 / (24 * 60)
    max_hl = 274.

    df['p'] = df['p_recall'].apply(lambda x: np.clip(float(x), 0.0001, .9999))
    df['t'] = df['delta'].apply(lambda x: float(x) / (60 * 60 * 24))
    df['h'] = df.apply(lambda row: np.clip(-row['t'] / np.log2(row['p']), min_hl, max_hl), axis=1)
    df['lang'] = df.apply(lambda row: f"{row['ui_language']} -> {row['learning_language']}", axis=1)
    df['lexeme'] = df.apply(lambda row: f"{row['learning_language']}:{row['lexeme_string']}", axis=1)
    df['right'] = df['history_correct'].astype(int)
    df['wrong'] = df['history_seen'].astype(int) - df['right']
    df['right_this'] = df['session_correct'].astype(int)
    df['wrong_this'] = df['session_seen'].astype(int) - df['right_this']
    df['right'] = df['right'].apply(lambda x: np.sqrt(1 + x))
    df['wrong'] = df['wrong'].apply(lambda x: np.sqrt(1 + x))
    df['bias'] = 1

    # Seconds to date
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Function to convert time to decimal hours
    def time_to_decimal(time):
        return time.hour + time.minute / 60 + time.second / 3600

    # Apply the function to the 'datetime' column
    df['decimal_hours'] = df['datetime'].apply(lambda x: time_to_decimal(x))

    # create the year var
    df['date_only'] = pd.to_datetime(df['datetime']).dt.date

    if not omit_lexemes:

        lexeme_dummies = pd.get_dummies(df['lexeme'], dtype=float)
        lexeme_columns = lexeme_dummies.columns.to_list()
        df = pd.concat([df, lexeme_dummies], axis=1)

        feature_vars = ['right', 'wrong', 'bias'] + lexeme_columns

    else:

        feature_vars = ['right', 'wrong', 'bias']

    def categorize_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    # Apply the function to your DataFrame
    df['time_of_day'] = df['decimal_hours'].apply(categorize_time_of_day)

    # Get dummy variables
    time_of_day_dummies = pd.get_dummies(df['time_of_day'], prefix='time')

    # Join the dummies back to the original DataFrame or use them separately as needed
    df = df.join(time_of_day_dummies)

    #     # Keep users either in training or testing
    #     grouped = df.groupby('user_id')

    #     # Create a list to store user IDs for training and testing
    #     train_users = []
    #     test_users = []

    #     for user_id in grouped.groups.keys():
    #         if np.random.rand() < 0.8:  # 80% chance
    #             train_users.append(user_id)
    #         else:
    #             test_users.append(user_id)

    #     trainset = df[df['user_id'].isin(train_users)]
    #     testset = df[df['user_id'].isin(test_users)]

    #     splitpoint = int(0.9 * len(df))
    #     trainset   = df[ : splitpoint ]
    #     testset    = df[ splitpoint : ]

    return df, feature_vars

clean_data, feature_vars = read_data( result_df_en_top20, omit_lexemes = False )


from sklearn.model_selection import train_test_split

# Assuming clean_data is your DataFrame, Y is the target column, and feature_vars are the features
# Extract unique users
unique_users = clean_data['user_id'].unique()  # Replace 'user_id' with your user identifier column

# Split users into train and test groups
train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)  # Adjust test_size as needed

# Create training and testing datasets
train_data = clean_data[clean_data['user_id'].isin(train_users)]
test_data = clean_data[clean_data['user_id'].isin(test_users)]


def create_sequences_optimized(data, feature_vars, time_steps=30):
    # Convert feature columns to numpy for faster processing
    feature_data = data[feature_vars].to_numpy()
    output_data = data['p'].to_numpy()  # Assuming 'p' is your target variable

    sequences = []
    outputs = []

    # Get the starting index for each new user
    user_change_indices = np.where(data['user_id'].to_numpy()[:-1] != data['user_id'].to_numpy()[1:])[0] + 1
    user_start_indices = np.insert(user_change_indices, 0, 0)  # Add 0 at the beginning

    # Iterate over each user
    for start_idx in user_start_indices:
        end_idx = start_idx + time_steps
        while end_idx < data.shape[0] and (end_idx not in user_change_indices):
            sequences.append(feature_data[start_idx:end_idx])
            outputs.append(output_data[end_idx])
            start_idx += 1
            end_idx += 1

    return np.array(sequences), np.array(outputs)

# Pre-sort the data by 'user_id' and 'datetime'
train_data_sorted = train_data.sort_values(by=['user_id', 'datetime'])
test_data_sorted = test_data.sort_values(by=['user_id', 'datetime'])

# Create training and testing sequences
X_train, Y_train = create_sequences_optimized(train_data_sorted, feature_vars)
X_test, Y_test = create_sequences_optimized(test_data_sorted, feature_vars)
