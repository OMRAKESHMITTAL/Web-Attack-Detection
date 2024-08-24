import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

def drop_non_numeric_columns(df):
    # Identify columns with data type 'object'
    object_columns = df.select_dtypes(include=['object']).columns
    # Drop columns with non-numeric data types
    df = df.drop(columns=object_columns)
    return df

# Function to load datasets
def load_data(paths):
    dataframes = [pd.read_csv(path, dtype=str, low_memory=False) for path in paths]
    return dataframes


# Function to clean data
def clean_data(df_list):
    for df in df_list:
        if 'Label' in df.columns:
            df.drop(df[df['Label'] == 'Label'].index, inplace=True)
    return df_list


# Function to perform stratified sampling
def stratified_sampling(dataframes, sample_size=10000):
    sampled_dfs = []
    for df in dataframes:
        if 'Label' in df.columns:
            grouped = df.groupby('Label', group_keys=False)
            sampled = grouped.apply(lambda x: x.sample(n=min(len(x), sample_size), replace=False), include_groups=False)
            sampled_dfs.append(sampled.reset_index(drop=True))
    return sampled_dfs


# Function to concatenate datasets
def concatenate_datasets(df_list):
    if not isinstance(df_list, list) or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise TypeError("Input must be a list of DataFrames")
    return pd.concat(df_list, ignore_index=True)


# Function to preprocess the dataset
def preprocess_data(df):
    # Convert Timestamp to int64
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', dayfirst=True).astype(np.int64)

    # Replace inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Fill remaining NaNs with 0
    df.fillna(0, inplace=True)

    # Convert 'Label' to numeric values
    label_mapping = {
        'Infilteration': 0, 'Bot': 0, 'DoS attacks-GoldenEye': 0, 'DoS attacks-Hulk': 0,
        'DoS attacks-Slowloris': 0, 'SSH-Bruteforce': 0, 'FTP-BruteForce': 0, 'DDOS attack-HOIC': 0,
        'DoS attacks-SlowHTTPTest': 0, 'DDOS attack-LOIC-UDP': 0, 'Brute Force -Web': 0,
        'Brute Force -XSS': 0, 'SQL Injection': 0, 'Benign': 1
    }
    df['Label'] = df['Label'].map(label_mapping)

    # Re-check for any infinity or NaN values and handle them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Replace excessively large values with random numbers
    def replace_large_values(df, threshold=1e10):
        numeric_cols = df.select_dtypes(include=[np.number])
        random_range = (-1e10, 1e10)  # Adjust as needed
        for col in numeric_cols.columns:
            df[col] = np.where(df[col] > threshold, np.random.uniform(*random_range, size=df[col].shape), df[col])
        return df

    df = replace_large_values(df)

    return df


# Function to drop constant features
def drop_constant_features(df, constant_features):
    return df.drop(columns=constant_features, axis=1)


# Function to identify and drop highly correlated features
def drop_highly_correlated_features(df, threshold=0.95):
    numeric_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr()
    correlated_features = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_features.add(corr_matrix.columns[i])

    return df.drop(columns=correlated_features, axis=1)


# Function to check for problematic values
def check_for_problems(df):
    # Check for NaNs in the dataset
    if df.isnull().values.any():
        print("Data contains NaNs")

    # Check for infinity values in numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if np.any(np.isinf(numeric_df.values)):
        print("Data contains infinity values")

    # Check for excessively large values
    if (numeric_df.max().max() > np.finfo(np.float32).max):
        print("Data contains values too large for dtype('float32')")

    print("Data integrity check complete.")


# Function to train and save the model
def train_and_save_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save the model to a .pkl file
    model_filename = 'xgb_model.pkl'
    joblib.dump(model, model_filename)
    print(f'Model saved to {model_filename}')

    # Evaluate the model
    predictions = model.predict(X_test)
    print(f'XGBClassifier Accuracy: {accuracy_score(y_test, predictions)}')
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for XGBClassifier')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# Paths to the datasets
paths = [
    '02-14-2018.csv',
    '02-15-2018.csv',
    '02-16-2018.csv',
    '02-21-2018.csv',
    '02-22-2018.csv',
    '02-23-2018.csv',
    '03-01-2018.csv',
    '03-02-2018.csv'
]

# Load and clean data
dataframes = load_data(paths)
print("Columns in loaded dataframes:")
for df in dataframes:
    print(df.columns)

dataframes = clean_data(dataframes)

# Verify if 'Label' column is still present after cleaning
for df in dataframes:
    if 'Label' not in df.columns:
        print("Warning: 'Label' column missing after cleaning.")
        print(df.columns)
    else:
        print("DataFrame with 'Label' column present.")

# Stratified sampling
sampled_data = stratified_sampling(dataframes[:3])
sampled_data = [df for df in sampled_data if isinstance(df, pd.DataFrame)]
sampled_data = pd.concat([df.head(10000) for df in dataframes[3:6]] + sampled_data, ignore_index=True)
sampled_data = stratified_sampling(dataframes[6:], sample_size=10000)
final_dataset = pd.concat(sampled_data, ignore_index=True)
file1 = '02-14-2018.csv'
df1 = pd.read_csv(file1)
column_to_add = 'Label'
column_data = df1[[column_to_add]].head(40000)
final_dataset = final_dataset.head(40000)
df2 = final_dataset
final_dataset = pd.concat([df2.reset_index(drop=True), column_data.reset_index(drop=True)], axis=1)
final_dataset.to_csv('updated_second_csv_file.csv', index=False)

# Preprocess data
print("Columns in final dataset before preprocessing:")
print(final_dataset.columns)
print(final_dataset)

final_dataset = preprocess_data(final_dataset)
check_for_problems(final_dataset)

# Drop constant features
constant_features = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
                     'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']
final_dataset = drop_constant_features(final_dataset, constant_features)

# Drop highly correlated features
final_dataset = drop_highly_correlated_features(final_dataset)
final_dataset = drop_non_numeric_columns(final_dataset)
final_dataset = final_dataset.head(1000)

# Prepare data for modeling
y = final_dataset['Label']
X = final_dataset.drop(columns=['Label'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Train and save the model
train_and_save_model(X_train, X_test, y_train, y_test)
