### Load Data
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Notebooks/

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a list of group names
group_names = ["wisdm", "pamap2", "daliac", "motionsense"]

# Create an empty dictionary to store DataFrames for each group
group_dataframes = {}

# Iterate over the groups
for group_name in group_names:
    # Load the dataframes for each group
    df_train = pd.read_csv(f"data/{group_name}_train.csv")
    df_test = pd.read_csv(f"data/{group_name}_test.csv")
    df_unseen = pd.read_csv(f"data/{group_name}_unseen.csv")

    # Store the DataFrames in the dictionary
    group_dataframes[group_name] = {
        "Train": df_train,
        "Test": df_test,
        "UnseenData": df_unseen
    }

    # Print information about the loaded data
    print(f"Loaded data for {group_name} group:")
    print(f"Train data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    print(f"Unseen data shape: {df_unseen.shape}")


    # Print info and descriptive statistics for each feature
    print("\nInfo for Train data:")
    print(df_train.info())
    print("\nDescriptive statistics for Train data:")
    #print(df_train.describe())
    print("\nInfo for Test data:")
    print(df_test.info())
    print("\nDescriptive statistics for Test data:")
    #print(df_test.describe())
    print("\nInfo for Unseen data:")
    print(df_unseen.info())
    print("\nDescriptive statistics for Unseen data:")
    #print(df_unseen.describe())

# Now, you can access the DataFrames for each group like this:
wisdm_df_train = group_dataframes["wisdm"]["Train"]
wisdm_df_test = group_dataframes["wisdm"]["Test"]
wisdm_df_unseen = group_dataframes["wisdm"]["UnseenData"]

# Similarly, for other groups:
pamap2_df_train = group_dataframes["pamap2"]["Train"]
pamap2_df_test = group_dataframes["pamap2"]["Test"]
pamap2_df_unseen = group_dataframes["pamap2"]["UnseenData"]

daliac_df_train = group_dataframes["daliac"]["Train"]
daliac_df_test = group_dataframes["daliac"]["Test"]
daliac_df_unseen = group_dataframes["daliac"]["UnseenData"]

motionsense_df_train = group_dataframes["motionsense"]["Train"]
motionsense_df_test = group_dataframes["motionsense"]["Test"]
motionsense_df_unseen = group_dataframes["motionsense"]["UnseenData"]


# Define the dataset names and splits
datasets = ["wisdm", "pamap2", "daliac", "motionsense"]
splits = ["Train", "Test", "UnseenData"]

# Iterate through datasets and splits
for dataset in datasets:
    for split in splits:
        # Create the dataframe variable name dynamically
        df_variable_name = f"{dataset}_df_{split.lower()}"

        # Access the dataframe using the dynamically created variable name
        current_df = group_dataframes[dataset][split]

        # Print summary
        print(f"Summary for {dataset} {split} Data:")
        print(current_df.info())
        print("\n" + "="*40 + "\n")  # Separating summaries with a line

"""Now, you can access the DataFrames for each group like this:
1. * wisdm_df_train
  * wisdm_df_test
  * wisdm_df_unseen

2. * pamap2_df_train
  * pamap2_df_test
  * pamap2_df_unseen
3.* daliac_df_train
  * daliac_df_test
  * daliac_df_unseen
4. * motionsense_df_train
  * motionsense_df_test
  * motionsense_df_unseen

unique_activities = ['walking', 'standing', 'sitting', 'upstairs', 'downstairs', 'running']

#2. adding timestamp for datasets
new 'timestamp' column added based on the sampling rate and reset when the activity changes:
"""

import pandas as pd

def add_timestamp(df, sampling_rate):
    # Calculate the time delta in seconds based on the sampling rate
    time_delta = pd.to_timedelta(df.groupby(['SubjectID', 'activity']).cumcount() * (1 / sampling_rate), unit='s')

    # Convert to a datetime series starting from an arbitrary fixed point (e.g., 2024-01-01)
    start_time = pd.Timestamp('2024-01-01')
    df['timestamp'] = start_time + time_delta
    return df

# Sampling rates (assuming the same sampling rate for all datasets; modify if different)
sampling_rate = 20

# 1. WISDM dataset
wisdm_df_train = add_timestamp(wisdm_df_train, sampling_rate)
wisdm_df_test = add_timestamp(wisdm_df_test, sampling_rate)
wisdm_df_unseen = add_timestamp(wisdm_df_unseen, sampling_rate)

# 2. PAMAP2 dataset
pamap2_df_train = add_timestamp(pamap2_df_train, sampling_rate)
pamap2_df_test = add_timestamp(pamap2_df_test, sampling_rate)
pamap2_df_unseen = add_timestamp(pamap2_df_unseen, sampling_rate)

# 3. DALIAC dataset
daliac_df_train = add_timestamp(daliac_df_train, sampling_rate)
daliac_df_test = add_timestamp(daliac_df_test, sampling_rate)
daliac_df_unseen = add_timestamp(daliac_df_unseen, sampling_rate)

# 4. MotionSense dataset
motionsense_df_train = add_timestamp(motionsense_df_train, sampling_rate)
motionsense_df_test = add_timestamp(motionsense_df_test, sampling_rate)
motionsense_df_unseen = add_timestamp(motionsense_df_unseen, sampling_rate)

wisdm_df_train

# Check for null values in each dataframe
print("Null values in wisdm_df_train:")
print(wisdm_df_train.isnull().sum())
print("Null values in wisdm_df_test:")
print(wisdm_df_test.isnull().sum())
print("Null values in wisdm_df_unseen:")
print(wisdm_df_unseen.isnull().sum())

print("Null values in pamap2_df_train:")
print(pamap2_df_train.isnull().sum())
print("Null values in pamap2_df_test:")
print(pamap2_df_test.isnull().sum())
print("Null values in pamap2_df_unseen:")
print(pamap2_df_unseen.isnull().sum())

print("Null values in daliac_df_train:")
print(daliac_df_train.isnull().sum())
print("Null values in daliac_df_test:")
print(daliac_df_test.isnull().sum())
print("Null values in daliac_df_unseen:")
print(daliac_df_unseen.isnull().sum())

print("Null values in motionsense_df_train:")
print(motionsense_df_train.isnull().sum())
print("Null values in motionsense_df_test:")
print(motionsense_df_test.isnull().sum())
print("Null values in motionsense_df_unseen:")
print(motionsense_df_unseen.isnull().sum())

# Clear Null
pamap2_df_train.dropna(inplace=True)
pamap2_df_test.dropna(inplace=True)
pamap2_df_unseen.dropna(inplace=True)

print("Null values in pamap2_df_train:")
print(pamap2_df_train.isnull().sum())
print("Null values in pamap2_df_test:")
print(pamap2_df_test.isnull().sum())
print("Null values in pamap2_df_unseen:")
print(pamap2_df_unseen.isnull().sum())

"""# 2.2 Undersampling From Raw"""

def balance_dataframe(df, activity_column='activity'):
    # Find the minimum number of samples for any activity
    min_samples = df[activity_column].value_counts().min()

    # Balance the dataframe by undersampling each activity
    balanced_df = df.groupby(activity_column).apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)
    return balanced_df

# 1. WISDM dataset
balanced_wisdm_df_train = balance_dataframe(wisdm_df_train)

# 2. PAMAP2 dataset
balanced_pamap2_df_train = balance_dataframe(pamap2_df_train)

# 3. DALIAC dataset
balanced_daliac_df_train = balance_dataframe(daliac_df_train)

# 4. MotionSense dataset
balanced_motionsense_df_train = balance_dataframe(motionsense_df_train)

# Check the balance of the categories
print(balanced_wisdm_df_train['activity'].value_counts())
print(balanced_pamap2_df_train['activity'].value_counts())
print(balanced_daliac_df_train['activity'].value_counts())
print(balanced_motionsense_df_train['activity'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

def plot_activity_distribution(original_df, balanced_df, dataset_name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

    # Plot original dataframe
    sns.countplot(ax=axes[0], x='activity', data=original_df, order=original_df['activity'].value_counts().index)
    axes[0].set_title(f'{dataset_name} - Before Balancing')
    axes[0].set_xlabel('Activity')
    axes[0].set_ylabel('Count')

    # Plot balanced dataframe
    sns.countplot(ax=axes[1], x='activity', data=balanced_df, order=balanced_df['activity'].value_counts().index)
    axes[1].set_title(f'{dataset_name} - After Balancing')
    axes[1].set_xlabel('Activity')

    plt.tight_layout()
    plt.show()

# Define the datasets
datasets = {
    'WISDM': (wisdm_df_train, balanced_wisdm_df_train),
    'PAMAP2': (pamap2_df_train, balanced_pamap2_df_train),
    'DALIAC': (daliac_df_train, balanced_daliac_df_train),
    'MotionSense': (motionsense_df_train, balanced_motionsense_df_train)
}

# Plot the distributions
for name, (original_df, balanced_df) in datasets.items():
    plot_activity_distribution(original_df, balanced_df, name)

"""### 3. Visulize By dataset group by activity"""

# import matplotlib.pyplot as plt
# wisdm
# Define activities and their counts for each group
activities_train = balanced_wisdm_df_train['activity'].value_counts()
activities_test = wisdm_df_test['activity'].value_counts()
activities_unseen = wisdm_df_unseen['activity'].value_counts()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
activities_train.plot(kind='barh', color='skyblue')
plt.title('Train Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 2)
activities_test.plot(kind='barh', color='lightgreen')
plt.title('Test Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 3)
activities_unseen.plot(kind='barh', color='salmon')
plt.title('Unseen Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# pamap2
# Define activities and their counts for each group
activities_train = balanced_pamap2_df_train['activity'].value_counts()
activities_test = pamap2_df_test['activity'].value_counts()
activities_unseen = pamap2_df_unseen['activity'].value_counts()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
activities_train.plot(kind='barh', color='skyblue')
plt.title('Train Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 2)
activities_test.plot(kind='barh', color='lightgreen')
plt.title('Test Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 3)
activities_unseen.plot(kind='barh', color='salmon')
plt.title('Unseen Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# daliac
# Define activities and their counts for each group
activities_train = balanced_daliac_df_train['activity'].value_counts()
activities_test = daliac_df_test['activity'].value_counts()
activities_unseen = daliac_df_unseen['activity'].value_counts()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
activities_train.plot(kind='barh', color='skyblue')
plt.title('Train Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 2)
activities_test.plot(kind='barh', color='lightgreen')
plt.title('Test Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 3)
activities_unseen.plot(kind='barh', color='salmon')
plt.title('Unseen Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# motionsense
# Define activities and their counts for each group
activities_train = balanced_motionsense_df_train['activity'].value_counts()
activities_test = motionsense_df_test['activity'].value_counts()
activities_unseen = motionsense_df_unseen['activity'].value_counts()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
activities_train.plot(kind='barh', color='skyblue')
plt.title('Train Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 2)
activities_test.plot(kind='barh', color='lightgreen')
plt.title('Test Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.subplot(3, 1, 3)
activities_unseen.plot(kind='barh', color='salmon')
plt.title('Unseen Group')
plt.xlabel('Number of Samples')
plt.ylabel('Activity')

plt.tight_layout()
plt.show()

"""#3. Feature Extration
 from the time domain for the columns [accX, accY, accZ] for each group, you can compute the mean, median, amplitude, and signal magnitude area (SMA)
 * Run this if need to use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_time_domain_features(df, window_size, sampling_rate):
    new_df = pd.DataFrame()

    for subject, activity_df in df.groupby(['SubjectID', 'activity']):
        activity_df = activity_df.sort_values(by='timestamp')

        window_size_rows = int(window_size * sampling_rate)

        for i in range(0, len(activity_df) - window_size_rows + 1, window_size_rows):
            window_data = activity_df.iloc[i:i + window_size_rows]

            mean_values = window_data[['accX', 'accY', 'accZ']].mean().values
            median_values = window_data[['accX', 'accY', 'accZ']].median().values
            amplitude_values = np.max(window_data[['accX', 'accY', 'accZ']].values, axis=0) - np.min(
                window_data[['accX', 'accY', 'accZ']].values, axis=0)
            sma_value = np.sum(np.abs(window_data[['accX', 'accY', 'accZ']].values))
            timestamp = window_data.iloc[0]['timestamp']

            # Create a dictionary for each calculated feature
            features_dict = {
                'SubjectID': subject[0],
                'activity': subject[1],
                'timestamp': timestamp,
                'Mean_accX': mean_values[0], 'Mean_accY': mean_values[1], 'Mean_accZ': mean_values[2],
                'Med_accX': median_values[0], 'Med_accY': median_values[1], 'Med_accZ': median_values[2],
                'Amp_accX': amplitude_values[0], 'Amp_accY': amplitude_values[1], 'Amp_accZ': amplitude_values[2],
                'SMA_accX': sma_value, 'SMA_accY': sma_value, 'SMA_accZ': sma_value
            }

            new_df = pd.concat([new_df, pd.DataFrame([features_dict])], ignore_index=True)

    new_df.reset_index(drop=True, inplace=True)
    return new_df

def balance_dataframe(df, activity_column='activity'):
    # Find the minimum number of samples for any activity
    min_samples = df[activity_column].value_counts().min()

    # Balance the dataframe by undersampling each activity
    balanced_df = df.groupby(activity_column).apply(lambda x: x.sample(min_samples)).reset_index(drop=True)
    return balanced_df

def plot_activity_distribution(original_df, balanced_df, dataset_name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

    # Plot original dataframe
    sns.countplot(ax=axes[0], x='activity', data=original_df, order=original_df['activity'].value_counts().index)
    axes[0].set_title(f'{dataset_name} - Before Balancing')
    axes[0].set_xlabel('Activity')
    axes[0].set_ylabel('Count')

    # Plot balanced dataframe
    sns.countplot(ax=axes[1], x='activity', data=balanced_df, order=balanced_df['activity'].value_counts().index)
    axes[1].set_title(f'{dataset_name} - After Balancing')
    axes[1].set_xlabel('Activity')

    plt.tight_layout()
    plt.show()

# Define parameters for feature extraction
window_size = 10  # in seconds
sampling_rate = 20  # in Hz, adjust according to your data

# Extract features from each training dataframe
extracted_wisdm_df_train = extract_time_domain_features(wisdm_df_train, window_size, sampling_rate)
extracted_pamap2_df_train = extract_time_domain_features(pamap2_df_train, window_size, sampling_rate)
extracted_daliac_df_train = extract_time_domain_features(daliac_df_train, window_size, sampling_rate)
extracted_motionsense_df_train = extract_time_domain_features(motionsense_df_train, window_size, sampling_rate)

# Balance the dataframes
balanced_extracted_wisdm_df_train = balance_dataframe(extracted_wisdm_df_train)
balanced_extracted_pamap2_df_train = balance_dataframe(extracted_pamap2_df_train)
balanced_extracted_daliac_df_train = balance_dataframe(extracted_daliac_df_train)
balanced_extracted_motionsense_df_train = balance_dataframe(extracted_motionsense_df_train)

# Plot the activity distribution before and after balancing for extracted data
plot_activity_distribution(extracted_wisdm_df_train, balanced_extracted_wisdm_df_train, 'WISDM')
plot_activity_distribution(extracted_pamap2_df_train, balanced_extracted_pamap2_df_train, 'PAMAP2')
plot_activity_distribution(extracted_daliac_df_train, balanced_extracted_daliac_df_train, 'DALIAC')
plot_activity_distribution(extracted_motionsense_df_train, balanced_extracted_motionsense_df_train, 'MotionSense')

# Extract features from each Test dataframe
wisdm_df_test_new = extract_time_domain_features(wisdm_df_test, window_size, sampling_rate)
pamap2_df_test_new = extract_time_domain_features(pamap2_df_test, window_size, sampling_rate)
daliac_df_test_new = extract_time_domain_features(daliac_df_test, window_size, sampling_rate)
motionsense_df_test_new = extract_time_domain_features(motionsense_df_test, window_size, sampling_rate)

# Check for null values in each dataframe
print("Null values in balanced_extracted_wisdm_df_train:")
print(balanced_extracted_wisdm_df_train.isnull().sum())
print("Null values in wisdm_df_test:")
print(wisdm_df_test_new.isnull().sum())
# print("Null values in wisdm_df_unseen:")
# print(wisdm_df_unseen.isnull().sum())

print("Null values in balanced_extracted_pamap2_df_train:")
print(balanced_extracted_pamap2_df_train.isnull().sum())
print("Null values in pamap2_df_test:")
print(pamap2_df_test_new.isnull().sum())
# print("Null values in pamap2_df_unseen:")
# print(pamap2_df_unseen.isnull().sum())

print("Null values in balanced_extracted_daliac_df_train:")
print(balanced_extracted_daliac_df_train.isnull().sum())
print("Null values in daliac_df_test:")
print(daliac_df_test_new.isnull().sum())
# print("Null values in daliac_df_unseen:")
# print(daliac_df_unseen.isnull().sum())

print("Null values in balanced_motionsense_df_train:")
print(balanced_extracted_motionsense_df_train.isnull().sum())
print("Null values in motionsense_df_test:")
print(motionsense_df_test_new.isnull().sum())
# print("Null values in motionsense_df_unseen:")
# print(motionsense_df_unseen.isnull().sum())

# Check for NaN values in each dataset
print("1. WISDM Dataset:")
print("   - Training Set NaNs:", balanced_extracted_wisdm_df_train.isna().sum().sum())
print("   - Test Set NaNs:", wisdm_df_test_new.isna().sum().sum())
# print("   - Unseen Set NaNs:", wisdm_df_unseen_new.isna().sum().sum())

print("2. PAMAP2 Dataset:")
print("   - Training Set NaNs:", balanced_extracted_pamap2_df_train.isna().sum().sum())
print("   - Test Set NaNs:", pamap2_df_test_new.isna().sum().sum())
# print("   - Unseen Set NaNs:", pamap2_df_unseen_new.isna().sum().sum())

print("3. Daliac Dataset:")
print("   - Training Set NaNs:", balanced_extracted_daliac_df_train.isna().sum().sum())
print("   - Test Set NaNs:", daliac_df_test_new.isna().sum().sum())
# print("   - Unseen Set NaNs:", daliac_df_unseen_new.isna().sum().sum())

print("4. Motionsense Dataset:")
print("   - Training Set NaNs:", balanced_extracted_daliac_df_train.isna().sum().sum())
print("   - Test Set NaNs:", motionsense_df_test_new.isna().sum().sum())
# print("   - Unseen Set NaNs:", motionsense_df_unseen_new.isna().sum().sum())

# Clear Null, If it has.
# pamap2_df_train.dropna(inplace=True)
# pamap2_df_test.dropna(inplace=True)
# pamap2_df_unseen.dropna(inplace=True)

# print("Null values in pamap2_df_train:")
# print(pamap2_df_train.isnull().sum())
# print("Null values in pamap2_df_test:")
# print(pamap2_df_test.isnull().sum())
# print("Null values in pamap2_df_unseen:")
# print(pamap2_df_unseen.isnull().sum())

# Print summary
print("Summary:")
print("1. Wisdm Dataset:")
print("   - Training Set Shape:", balanced_extracted_wisdm_df_train.shape)
print("   - Test Set Shape:", wisdm_df_test_new.shape)
# print("   - Unseen Set Shape:", wisdm_df_unseen_new.shape)

print("2. PAMAP2 Dataset:")
print("   - Training Set Shape:", balanced_extracted_pamap2_df_train.shape)
print("   - Test Set Shape:", pamap2_df_test_new.shape)
# print("   - Unseen Set Shape:", pamap2_df_unseen_new.shape)

print("3. Daliac Dataset:")
print("   - Training Set Shape:", balanced_extracted_daliac_df_train.shape)
print("   - Test Set Shape:", daliac_df_test_new.shape)
# print("   - Unseen Set Shape:", daliac_df_unseen_new.shape)

print("4. Motionsense Dataset:")
print("   - Training Set Shape:", balanced_extracted_motionsense_df_train.shape)
print("   - Test Set Shape:", motionsense_df_test_new.shape)
# print("   - Unseen Set Shape:", motionsense_df_unseen_new.shape)

# pamap2_df_train_new = pamap2_df_train_new.dropna()
# pamap2_df_test_new = pamap2_df_test_new.dropna()
# pamap2_df_unseen_new = pamap2_df_unseen_new.dropna()

# # Check shapes after removal
# print("PAMAP2 Dataset after removing NaNs:")
# print("   - Training Set Shape:", pamap2_df_train_new.shape)
# print("   - Test Set Shape:", pamap2_df_test_new.shape)
# print("   - Unseen Set Shape:", pamap2_df_unseen_new.shape)

extracted_wisdm_df_train

"""# 4. Normalize"""

from sklearn.preprocessing import StandardScaler

train_datasets = {
    'wisdm': balanced_extracted_wisdm_df_train,
    'pamap2': balanced_extracted_pamap2_df_train,
    'daliac': balanced_extracted_daliac_df_train,
    'motionsense': balanced_extracted_motionsense_df_train
}

test_datasets = {
    'wisdm': wisdm_df_test_new,
    'pamap2': pamap2_df_test_new,
    'daliac': daliac_df_test_new,
    'motionsense': motionsense_df_test_new
}

# Define the feature columns to normalize
feature_columns = [
    'Mean_accX', 'Mean_accY', 'Mean_accZ',
    'Med_accX', 'Med_accY', 'Med_accZ',
    'Amp_accX', 'Amp_accY', 'Amp_accZ',
    'SMA_accX', 'SMA_accY', 'SMA_accZ'
]

# Define a function to normalize the feature columns
def normalize_features(df):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def plot_histograms_before_after_normalization(df, feature_columns):
    fig, axes = plt.subplots(nrows=len(feature_columns), ncols=2, figsize=(14, 8))

    for i, column in enumerate(feature_columns):
        # Plot histogram before normalization
        axes[i, 0].hist(df[column], bins=30, color='blue', alpha=0.7)
        axes[i, 0].set_title(f'Before Normalization - {column}')
        axes[i, 0].set_xlabel('Value')
        axes[i, 0].set_ylabel('Frequency')

        # Plot histogram after normalization
        axes[i, 1].hist(df[column], bins=30, color='orange', alpha=0.7)
        axes[i, 1].set_title(f'After Normalization - {column}')
        axes[i, 1].set_xlabel('Value')
        axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Plot histograms for each dataframe before and after normalization
for dataset_name, df in train_datasets.items():
    print(f"Dataset: {dataset_name}")
    plot_histograms_before_after_normalization(df, feature_columns)

def check_activity_counts(datasets):
    for dataset_name, dataset_df in datasets.items():
        print(f"\nDataset: {dataset_name}")
        activity_counts = dataset_df['activity'].value_counts()
        print(activity_counts)

# Example usage
check_activity_counts(train_datasets)
check_activity_counts(test_datasets)

"""# Model"""

activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
}
train_datasets = {
    'wisdm': normalize_features(balanced_extracted_wisdm_df_train),
    'pamap2': normalize_features(balanced_extracted_pamap2_df_train),
    'daliac': normalize_features(balanced_extracted_daliac_df_train),
    'motionsense': normalize_features(balanced_extracted_motionsense_df_train)
}

test_datasets = {
    'wisdm': normalize_features(wisdm_df_test_new),
    'pamap2': normalize_features(pamap2_df_test_new),
    'daliac': normalize_features(daliac_df_test_new),
    'motionsense': normalize_features(motionsense_df_test_new)
}

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

def create_mlp_and_predict_all_targets_with_report(name, input_df_train, input_df_test):
    X_train = input_df_train[feature_columns]
    X_test = input_df_test[feature_columns]

    activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
    }
    input_df_train['activity_num'] = input_df_train['activity'].map(activity_map)
    input_df_test['activity_num'] = input_df_test['activity'].map(activity_map)
    y_train = input_df_train['activity_num']
    y_test = input_df_test['activity_num']

    mlp = MLPClassifier()

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    grid_search = GridSearchCV(mlp, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    softmax_values = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)

    print(f"Classification Report for {name} Dataset:")
    print(classification_report(y_test, y_pred, target_names=list(activity_map.keys())))
    print(f"Confusion Matrix for {name} Dataset:")
    print(confusion_matrix(y_test, y_pred))

    return {
        'name': name,
        'softmax': softmax_values,
        'accuracy': accuracy,
        'weights': best_model.coefs_,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def data_fusion(train_datasets, test_datasets):
    results = {}
    for name in train_datasets:
        result = create_mlp_and_predict_all_targets_with_report(name, train_datasets[name], test_datasets[name])
        results[name] = result

    min_length = min(result['softmax'].shape[0] for result in results.values())

    for name in results:
        softmax_values = results[name]['softmax']
        if softmax_values.shape[0] > min_length:
            results[name]['softmax'] = softmax_values[:min_length]
        elif softmax_values.shape[0] < min_length:
            pad_length = min_length - softmax_values.shape[0]
            results[name]['softmax'] = np.pad(softmax_values, ((0, pad_length), (0, 0)), 'constant')

    combined_softmax = np.hstack([results[name]['softmax'] for name in results])

    final_y_test = results[list(test_datasets.keys())[0]]['y_test'][:min_length]

    final_X_train, final_X_test, final_y_train, final_y_test = train_test_split(
        combined_softmax, final_y_test, test_size=0.2, random_state=42
    )

    final_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.001)
    final_mlp.fit(final_X_train, final_y_train)

    final_y_pred_train = final_mlp.predict(final_X_train)
    final_y_pred_test = final_mlp.predict(final_X_test)

    final_accuracy_train = accuracy_score(final_y_train, final_y_pred_train)
    final_accuracy_test = accuracy_score(final_y_test, final_y_pred_test)

    print("Final Model - Training Data")
    print(f"Accuracy: {final_accuracy_train}")
    print("Classification Report:")
    print(classification_report(final_y_train, final_y_pred_train, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_train, final_y_pred_train))

    print("Final Model - Test Data")
    print(f"Accuracy: {final_accuracy_test}")
    print("Classification Report:")
    print(classification_report(final_y_test, final_y_pred_test, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_test, final_y_pred_test))

    return final_mlp, final_y_test, final_y_pred_test

# Perform data fusion and print classification reports and confusion matrices
final_model, final_y_test, final_y_pred_test = data_fusion(train_datasets, test_datasets)

"""....END....

# Improvement The data fusion Step
"""

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np


def extract_time_domain_features(df, window_size, sampling_rate):
    new_df = pd.DataFrame()
    for subject, activity_df in df.groupby(['SubjectID', 'activity']):
        activity_df = activity_df.sort_values(by='timestamp')
        window_size_rows = int(window_size * sampling_rate)
        for i in range(0, len(activity_df) - window_size_rows + 1, window_size_rows):
            window_data = activity_df.iloc[i:i + window_size_rows]
            mean_values = window_data[['accX', 'accY', 'accZ']].mean().values
            median_values = window_data[['accX', 'accY', 'accZ']].median().values
            amplitude_values = np.max(window_data[['accX', 'accY', 'accZ']].values, axis=0) - np.min(
                window_data[['accX', 'accY', 'accZ']].values, axis=0)
            sma_value = np.sum(np.abs(window_data[['accX', 'accY', 'accZ']].values))
            timestamp = window_data.iloc[0]['timestamp']
            features_dict = {
                'SubjectID': subject[0],
                'activity': subject[1],
                'timestamp': timestamp,
                'Mean_accX': mean_values[0], 'Mean_accY': mean_values[1], 'Mean_accZ': mean_values[2],
                'Med_accX': median_values[0], 'Med_accY': median_values[1], 'Med_accZ': median_values[2],
                'Amp_accX': amplitude_values[0], 'Amp_accY': amplitude_values[1], 'Amp_accZ': amplitude_values[2],
                'SMA_accX': sma_value, 'SMA_accY': sma_value, 'SMA_accZ': sma_value
            }
            new_df = pd.concat([new_df, pd.DataFrame.from_dict([features_dict])], ignore_index=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def normalize_features(df):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

window_size = 5  # Example window size in seconds
sampling_rate = 20  # Example sampling rate in Hz
feature_columns = [
    'Mean_accX', 'Mean_accY', 'Mean_accZ', 'Med_accX', 'Med_accY', 'Med_accZ',
    'Amp_accX', 'Amp_accY', 'Amp_accZ', 'SMA_accX', 'SMA_accY', 'SMA_accZ'
]

# Balance, extract, and normalize the data
datasets = {
    'wisdm': (wisdm_df_train, wisdm_df_test),
    'pamap2': (pamap2_df_train, pamap2_df_test),
    'daliac': (daliac_df_train, daliac_df_test),
    'motionsense': (motionsense_df_train, motionsense_df_test)
}

# Extract and normalize features
normalized_datasets = {}
for name, (train_df, test_df) in datasets.items():
    extracted_train_df = extract_time_domain_features(train_df, window_size, sampling_rate)
    extracted_test_df = extract_time_domain_features(test_df, window_size, sampling_rate)
    normalized_datasets[name] = (
        normalize_features(extracted_train_df),
        normalize_features(extracted_test_df)
    )


def create_mlp_and_predict_all_targets_with_report(name, input_df_train, input_df_test):
    X_train = input_df_train[feature_columns]
    X_test = input_df_test[feature_columns]

    activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
    }
    input_df_train['activity_num'] = input_df_train['activity'].map(activity_map)
    input_df_test['activity_num'] = input_df_test['activity'].map(activity_map)
    y_train = input_df_train['activity_num']
    y_test = input_df_test['activity_num']

    mlp = MLPClassifier()
    rf = RandomForestClassifier()

    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, scoring='accuracy', cv=5)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, scoring='accuracy', cv=5)

    grid_search_mlp.fit(X_train, y_train)
    grid_search_rf.fit(X_train, y_train)

    best_mlp = grid_search_mlp.best_estimator_
    best_rf = grid_search_rf.best_estimator_

    voting_clf = VotingClassifier(estimators=[('mlp', best_mlp), ('rf', best_rf)], voting='soft')
    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    y_pred_proba = voting_clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    softmax_values = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)

    print(f"Classification Report for {name} Dataset:")
    print(classification_report(y_test, y_pred, target_names=list(activity_map.keys())))
    print(f"Confusion Matrix for {name} Dataset:")
    print(confusion_matrix(y_test, y_pred))

    return {
        'name': name,
        'softmax': softmax_values,
        'accuracy': accuracy,
        'weights': voting_clf,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def data_fusion(train_datasets, test_datasets):
    results = {}
    for name in train_datasets:
        result = create_mlp_and_predict_all_targets_with_report(name, train_datasets[name], test_datasets[name])
        results[name] = result

    min_length = min(result['softmax'].shape[0] for result in results.values())

    for name in results:
        softmax_values = results[name]['softmax']
        if softmax_values.shape[0] > min_length:
            results[name]['softmax'] = softmax_values[:min_length]
        elif softmax_values.shape[0] < min_length:
            pad_length = min_length - softmax_values.shape[0]
            results[name]['softmax'] = np.pad(softmax_values, ((0, pad_length), (0, 0)), 'constant')

    combined_softmax = np.hstack([results[name]['softmax'] for name in results])

    final_y_test = results[list(test_datasets.keys())[0]]['y_test'][:min_length]

    final_X_train, final_X_test, final_y_train, final_y_test = train_test_split(
        combined_softmax, final_y_test, test_size=0.2, random_state=42
    )

    final_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.001)
    final_mlp.fit(final_X_train, final_y_train)

    final_y_pred_train = final_mlp.predict(final_X_train)
    final_y_pred_test = final_mlp.predict(final_X_test)

    final_accuracy_train = accuracy_score(final_y_train, final_y_pred_train)
    final_accuracy_test = accuracy_score(final_y_test, final_y_pred_test)

    print("Final Model - Training Data")
    print(f"Accuracy: {final_accuracy_train}")
    print("Classification Report:")
    print(classification_report(final_y_train, final_y_pred_train, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_train, final_y_pred_train))

    print("Final Model - Test Data")
    print(f"Accuracy: {final_accuracy_test}")
    print("Classification Report:")
    print(classification_report(final_y_test, final_y_pred_test, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_test, final_y_pred_test))

    return final_mlp, final_y_test, final_y_pred_test

# Balance, extract, and normalize the data
extracted_wisdm_df_train = normalize_features(extract_time_domain_features(wisdm_df_train, window_size, sampling_rate))
extracted_pamap2_df_train = normalize_features(extract_time_domain_features(pamap2_df_train, window_size, sampling_rate))
extracted_daliac_df_train = normalize_features(extract_time_domain_features(daliac_df_train, window_size, sampling_rate))
extracted_motionsense_df_train = normalize_features(extract_time_domain_features(motionsense_df_train, window_size, sampling_rate))

wisdm_df_test_new = normalize_features(extract_time_domain_features(wisdm_df_test, window_size, sampling_rate))
pamap2_df_test_new = normalize_features(extract_time_domain_features(pamap2_df_test, window_size, sampling_rate))
daliac_df_test_new = normalize_features(extract_time_domain_features(daliac_df_test, window_size, sampling_rate))
motionsense_df_test_new = normalize_features(extract_time_domain_features(motionsense_df_test, window_size, sampling_rate))

activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
}

train_datasets = {
    'wisdm': extracted_wisdm_df_train,
    'pamap2': extracted_pamap2_df_train,
    'daliac': extracted_daliac_df_train,
    'motionsense': extracted_motionsense_df_train
}

test_datasets = {
    'wisdm': wisdm_df_test_new,
    'pamap2': pamap2_df_test_new,
    'daliac': daliac_df_test_new,
    'motionsense': motionsense_df_test_new
}

# Perform data fusion and print classification reports and confusion matrices
final_model, final_y_test, final_y_pred_test = data_fusion(train_datasets, test_datasets)

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import warnings

# Import the relevant functions/classes
from sklearn.exceptions import ConvergenceWarning

# Filter out the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def extract_time_domain_features(df, window_size, sampling_rate):
    new_df = pd.DataFrame()
    for subject, activity_df in df.groupby(['SubjectID', 'activity']):
        activity_df = activity_df.sort_values(by='timestamp')
        window_size_rows = int(window_size * sampling_rate)
        for i in range(0, len(activity_df) - window_size_rows + 1, window_size_rows):
            window_data = activity_df.iloc[i:i + window_size_rows]
            mean_values = window_data[['accX', 'accY', 'accZ']].mean().values
            median_values = window_data[['accX', 'accY', 'accZ']].median().values
            amplitude_values = np.max(window_data[['accX', 'accY', 'accZ']].values, axis=0) - np.min(
                window_data[['accX', 'accY', 'accZ']].values, axis=0)
            sma_value = np.sum(np.abs(window_data[['accX', 'accY', 'accZ']].values))
            timestamp = window_data.iloc[0]['timestamp']
            features_dict = {
                'SubjectID': subject[0],
                'activity': subject[1],
                'timestamp': timestamp,
                'Mean_accX': mean_values[0], 'Mean_accY': mean_values[1], 'Mean_accZ': mean_values[2],
                'Med_accX': median_values[0], 'Med_accY': median_values[1], 'Med_accZ': median_values[2],
                'Amp_accX': amplitude_values[0], 'Amp_accY': amplitude_values[1], 'Amp_accZ': amplitude_values[2],
                'SMA_accX': sma_value, 'SMA_accY': sma_value, 'SMA_accZ': sma_value
            }
            new_df = pd.concat([new_df, pd.DataFrame.from_dict([features_dict])], ignore_index=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def normalize_features(df):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

window_size = 5  # Example window size in seconds
sampling_rate = 20  # Example sampling rate in Hz
feature_columns = [
    'Mean_accX', 'Mean_accY', 'Mean_accZ', 'Med_accX', 'Med_accY', 'Med_accZ',
    'Amp_accX', 'Amp_accY', 'Amp_accZ', 'SMA_accX', 'SMA_accY', 'SMA_accZ'
]

# Balance, extract, and normalize the data
datasets = {
    'wisdm': (wisdm_df_train, wisdm_df_test),
    'pamap2': (pamap2_df_train, pamap2_df_test),
    'daliac': (daliac_df_train, daliac_df_test),
    'motionsense': (motionsense_df_train, motionsense_df_test)
}

# Extract and normalize features
normalized_datasets = {}
for name, (train_df, test_df) in datasets.items():
    extracted_train_df = extract_time_domain_features(train_df, window_size, sampling_rate)
    extracted_test_df = extract_time_domain_features(test_df, window_size, sampling_rate)
    normalized_datasets[name] = (
        normalize_features(extracted_train_df),
        normalize_features(extracted_test_df)
    )


def create_mlp_and_predict_all_targets_with_report(name, input_df_train, input_df_test):
    X_train = input_df_train[feature_columns]
    X_test = input_df_test[feature_columns]

    activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
    }
    input_df_train['activity_num'] = input_df_train['activity'].map(activity_map)
    input_df_test['activity_num'] = input_df_test['activity'].map(activity_map)
    y_train = input_df_train['activity_num']
    y_test = input_df_test['activity_num']

    mlp = MLPClassifier()
    rf = RandomForestClassifier()

    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, scoring='accuracy', cv=5)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, scoring='accuracy', cv=5)

    grid_search_mlp.fit(X_train, y_train)
    grid_search_rf.fit(X_train, y_train)

    best_mlp = grid_search_mlp.best_estimator_
    best_rf = grid_search_rf.best_estimator_

    voting_clf = VotingClassifier(estimators=[('mlp', best_mlp), ('rf', best_rf)], voting='soft')
    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    y_pred_proba = voting_clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    softmax_values = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)

    print(f"Classification Report for {name} Dataset:")
    print(classification_report(y_test, y_pred, target_names=list(activity_map.keys())))
    print(f"Confusion Matrix for {name} Dataset:")
    print(confusion_matrix(y_test, y_pred))

    return {
        'name': name,
        'softmax': softmax_values,
        'accuracy': accuracy,
        'weights': voting_clf,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def data_fusion(train_datasets, test_datasets):
    results = {}
    classification_reports = {}
    confusion_matrices = {}

    for name in train_datasets:
        # Train and test each dataset individually
        result = create_mlp_and_predict_all_targets_with_report(name, train_datasets[name], test_datasets[name])
        results[name] = result

        # Store classification report and confusion matrix
        y_test = result['y_test']
        y_pred = result['y_pred']
        activity_map = {
            'downstairs': 1,
            'upstairs': 2,
            'walking': 3,
            'running': 4,
            'standing': 5,
            'sitting': 6
        }
        classification_reports[name] = classification_report(y_test, y_pred, target_names=list(activity_map.keys()))
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    # Combine results if needed for further fusion
    min_length = min(result['softmax'].shape[0] for result in results.values())
    combined_softmax = np.hstack([results[name]['softmax'][:min_length] for name in results])
    final_y_test = results[list(test_datasets.keys())[0]]['y_test'][:min_length]

    # Split the combined data for training and testing the final model
    final_X_train, final_X_test, final_y_train, final_y_test = train_test_split(
        combined_softmax, final_y_test, test_size=0.2, random_state=42
    )

    # Train the final model (MLP classifier)
    final_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.001)
    final_mlp.fit(final_X_train, final_y_train)

    # Predictions on the final model for training and test data
    final_y_pred_train = final_mlp.predict(final_X_train)
    final_y_pred_test = final_mlp.predict(final_X_test)

    # Calculate accuracies
    final_accuracy_train = accuracy_score(final_y_train, final_y_pred_train)
    final_accuracy_test = accuracy_score(final_y_test, final_y_pred_test)

    # Print results for training data
    print("Final Model - Training Data")
    print(f"Accuracy: {final_accuracy_train}")
    print("Classification Report:")
    print(classification_report(final_y_train, final_y_pred_train, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_train, final_y_pred_train))

    # Print results for test data
    print("Final Model - Test Data")
    print(f"Accuracy: {final_accuracy_test}")
    print("Classification Report:")
    print(classification_report(final_y_test, final_y_pred_test, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_test, final_y_pred_test))

    # Return final model, final test labels, and predictions
    return final_mlp, final_y_test, final_y_pred_test, classification_reports, confusion_matrices

# Balance, extract, and normalize the data
extracted_wisdm_df_train = normalize_features(extract_time_domain_features(wisdm_df_train, window_size, sampling_rate))
extracted_pamap2_df_train = normalize_features(extract_time_domain_features(pamap2_df_train, window_size, sampling_rate))
extracted_daliac_df_train = normalize_features(extract_time_domain_features(daliac_df_train, window_size, sampling_rate))
extracted_motionsense_df_train = normalize_features(extract_time_domain_features(motionsense_df_train, window_size, sampling_rate))

wisdm_df_test_new = normalize_features(extract_time_domain_features(wisdm_df_test, window_size, sampling_rate))
pamap2_df_test_new = normalize_features(extract_time_domain_features(pamap2_df_test, window_size, sampling_rate))
daliac_df_test_new = normalize_features(extract_time_domain_features(daliac_df_test, window_size, sampling_rate))
motionsense_df_test_new = normalize_features(extract_time_domain_features(motionsense_df_test, window_size, sampling_rate))

activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
}

train_datasets = {
    'wisdm': extracted_wisdm_df_train,
    'pamap2': extracted_pamap2_df_train,
    'daliac': extracted_daliac_df_train,
    'motionsense': extracted_motionsense_df_train
}

test_datasets = {
    'wisdm': wisdm_df_test_new,
    'pamap2': pamap2_df_test_new,
    'daliac': daliac_df_test_new,
    'motionsense': motionsense_df_test_new
}

# Perform data fusion and print classification reports and confusion matrices
final_model, final_y_test, final_y_pred_test = data_fusion(train_datasets, test_datasets)

# FIX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd

def extract_time_domain_features(df, window_size, sampling_rate):
    new_df = pd.DataFrame()
    for subject, activity_df in df.groupby(['SubjectID', 'activity']):
        activity_df = activity_df.sort_values(by='timestamp')
        window_size_rows = int(window_size * sampling_rate)
        for i in range(0, len(activity_df) - window_size_rows + 1, window_size_rows):
            window_data = activity_df.iloc[i:i + window_size_rows]
            mean_values = window_data[['accX', 'accY', 'accZ']].mean().values
            median_values = window_data[['accX', 'accY', 'accZ']].median().values
            amplitude_values = np.max(window_data[['accX', 'accY', 'accZ']].values, axis=0) - np.min(
                window_data[['accX', 'accY', 'accZ']].values, axis=0)
            sma_value = np.sum(np.abs(window_data[['accX', 'accY', 'accZ']].values))
            timestamp = window_data.iloc[0]['timestamp']
            features_dict = {
                'SubjectID': subject[0],
                'activity': subject[1],
                'timestamp': timestamp,
                'Mean_accX': mean_values[0], 'Mean_accY': mean_values[1], 'Mean_accZ': mean_values[2],
                'Med_accX': median_values[0], 'Med_accY': median_values[1], 'Med_accZ': median_values[2],
                'Amp_accX': amplitude_values[0], 'Amp_accY': amplitude_values[1], 'Amp_accZ': amplitude_values[2],
                'SMA_accX': sma_value, 'SMA_accY': sma_value, 'SMA_accZ': sma_value
            }
            new_df = pd.concat([new_df, pd.DataFrame.from_dict([features_dict])], ignore_index=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def normalize_features(df):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

window_size = 5  # Example window size in seconds
sampling_rate = 20  # Example sampling rate in Hz
feature_columns = [
    'Mean_accX', 'Mean_accY', 'Mean_accZ', 'Med_accX', 'Med_accY', 'Med_accZ',
    'Amp_accX', 'Amp_accY', 'Amp_accZ', 'SMA_accX', 'SMA_accY', 'SMA_accZ'
]

# Define your datasets without balancing
datasets = {
    'wisdm': (wisdm_df_train, wisdm_df_test),
    'pamap2': (pamap2_df_train, pamap2_df_test),
    'daliac': (daliac_df_train, daliac_df_test),
    'motionsense': (motionsense_df_train, motionsense_df_test)
}

# Extract and normalize features
normalized_datasets = {}
for name, (train_df, test_df) in datasets.items():
    extracted_train_df = extract_time_domain_features(train_df, window_size, sampling_rate)
    extracted_test_df = extract_time_domain_features(test_df, window_size, sampling_rate)
    normalized_datasets[name] = (
        normalize_features(extracted_train_df),
        normalize_features(extracted_test_df)
    )

def create_mlp_and_predict_all_targets_with_report(name, input_df_train, input_df_test):
    X_train = input_df_train[feature_columns]
    X_test = input_df_test[feature_columns]

    activity_map = {
        'downstairs': 1,
        'upstairs': 2,
        'walking': 3,
        'running': 4,
        'standing': 5,
        'sitting': 6
    }
    input_df_train['activity_num'] = input_df_train['activity'].map(activity_map)
    input_df_test['activity_num'] = input_df_test['activity'].map(activity_map)
    y_train = input_df_train['activity_num']
    y_test = input_df_test['activity_num']

    mlp = MLPClassifier()
    rf = RandomForestClassifier()

    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, scoring='accuracy', cv=5)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, scoring='accuracy', cv=5)

    grid_search_mlp.fit(X_train, y_train)
    grid_search_rf.fit(X_train, y_train)

    best_mlp = grid_search_mlp.best_estimator_
    best_rf = grid_search_rf.best_estimator_

    voting_clf = VotingClassifier(estimators=[('mlp', best_mlp), ('rf', best_rf)], voting='soft')
    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    y_pred_proba = voting_clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    softmax_values = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)

    print(f"Classification Report for {name} Dataset:")
    print(classification_report(y_test, y_pred, target_names=list(activity_map.keys())))
    print(f"Confusion Matrix for {name} Dataset:")
    print(confusion_matrix(y_test, y_pred))

    return {
        'name': name,
        'softmax': softmax_values,
        'accuracy': accuracy,
        'weights': voting_clf,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'estimators': voting_clf.estimators_
    }

def data_fusion(train_datasets, test_datasets):
    results = {}
    for name in train_datasets:
        result = create_mlp_and_predict_all_targets_with_report(name, train_datasets[name], test_datasets[name])
        results[name] = result

    min_length = min(result['softmax'].shape[0] for result in results.values())

    for name in results:
        softmax_values = results[name]['softmax']
        if softmax_values.shape[0] > min_length:
            results[name]['softmax'] = softmax_values[:min_length]
        elif softmax_values.shape[0] < min_length:
            pad_length = min_length - softmax_values.shape[0]
            results[name]['softmax'] = np.pad(softmax_values, ((0, pad_length), (0, 0)), 'constant')

    combined_softmax = np.hstack([results[name]['softmax'] for name in results])

    final_y_test = results[list(test_datasets.keys())[0]]['y_test'][:min_length]

    final_X_train, final_X_test, final_y_train, final_y_test = train_test_split(
        combined_softmax, final_y_test, test_size=0.2, random_state=42
    )

    final_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.001)
    final_mlp.fit(final_X_train, final_y_train)

    final_y_pred_train = final_mlp.predict(final_X_train)
    final_y_pred_test = final_mlp.predict(final_X_test)

    final_accuracy_train = accuracy_score(final_y_train, final_y_pred_train)
    final_accuracy_test = accuracy_score(final_y_test, final_y_pred_test)

    print("Final Model - Training Data")
    print(f"Accuracy: {final_accuracy_train}")
    print("Classification Report:")
    print(classification_report(final_y_train, final_y_pred_train, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_train, final_y_pred_train))

    print("Final Model - Test Data")
    print(f"Accuracy: {final_accuracy_test}")
    print("Classification Report:")
    print(classification_report(final_y_test, final_y_pred_test, target_names=list(activity_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(final_y_test, final_y_pred_test))

    return final_mlp, final_y_test, final_y_pred_test, results

# Define train and test datasets after normalization and feature extraction
train_datasets = {
    'wisdm': normalized_datasets['wisdm'][0],
    'pamap2': normalized_datasets['pamap2'][0],
    'daliac': normalized_datasets['daliac'][0],
    'motionsense': normalized_datasets['motionsense'][0]
}

test_datasets = {
    'wisdm': normalized_datasets['wisdm'][1],
    'pamap2': normalized_datasets['pamap2'][1],
    'daliac': normalized_datasets['daliac'][1],
    'motionsense': normalized_datasets['motionsense'][1]
}

# Perform data fusion and print classification reports and confusion matrices
final_model, final_y_test, final_y_pred_test, results = data_fusion(train_datasets, test_datasets)

# Print the winning estimators for each dataframe
for name in results:
    print(f"\n{name} Voting Classifier Estimators:")
    for estimator in results[name]['estimators']:
        print(estimator)

