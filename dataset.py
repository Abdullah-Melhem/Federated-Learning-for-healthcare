from collections import Counter
import torch
from sklearn.compose import ColumnTransformer
from torch.utils.data import random_split, DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
from category_encoders import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset


def preproccesing_fn(df):
    """prepare the HA-data set """

    df.drop(columns=['State', 'Sex'], inplace=True)
    replacement_dict = {'Yes': 1, 'No': 0}
    df['HadHeartAttack'] = df['HadHeartAttack'].replace(replacement_dict)
    df['HadAngina'] = df['HadAngina'].replace(replacement_dict)
    df['HeartDisease'] = df['HadHeartAttack'] | df['HadAngina']
    df.drop(columns=['HadHeartAttack', 'HadAngina'], inplace=True)

    df_cat = df.select_dtypes('object')
    df.drop(columns='WeightInKilograms', inplace=True)

    df_categorical = df_cat.columns
    preprocessor = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), df_categorical)],
                                     remainder='passthrough')

    x = preprocessor.fit_transform(df.drop(columns='HeartDisease'))
    y = df['HeartDisease']

    over = SMOTE(sampling_strategy=1)
    under = RandomUnderSampler(sampling_strategy=0.1)

    x_resembled, y_resembled = under.fit_resample(x, y)
    X, Y = over.fit_resample(x_resembled, y_resembled)

    print(f"\nX_train shape: {X.shape} y_train shape: {Y.shape}")
    print(f"Number of records for each class: {Counter(Y)}\n")

    return X, Y


def get_data(data_path, train_ratio):
    """Upload heart and apply minimal transformation."""

    df = pd.read_csv(data_path)

    features, labels = preproccesing_fn(df=df)

    features = torch.tensor(features, dtype=torch.float32)

    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(features, labels)

    num_samples = len(dataset)

    num_train_samples = int(train_ratio * num_samples)

    num_test_samples = num_samples - num_train_samples

    # Split the dataset into training and testing sets
    trainset, testset = random_split(dataset, [num_train_samples, num_test_samples],
                                     torch.Generator().manual_seed(2023))

    return trainset, testset



def print_label_counts(partitions, dataset,target_clt):
    """Print the count of each label in each partition, specifically for class 0 and class 1."""
    for i, partition in enumerate(partitions):
        # Extract labels from the subset using the indices
        labels = []
        for idx in (partition.indices if hasattr(partition, 'indices') else partition):
            try:
                # Convert tensor label to a Python integer
                label = dataset[idx][1].item()
                labels.append(label)
            except Exception as e:
                print(f"Error accessing dataset[{idx}]: {e}")

        # Count occurrences of each class
        count = Counter(labels)
        class_0_count = count.get(0, 0)  # Count for class 0
        class_1_count = count.get(1, 0)  # Count for class 1

        print(f"{target_clt} {i} - Class 0: {class_0_count}, Class 1: {class_1_count}")


def stratified_split(dataset, num_partitions, seed=2023):
    """Split dataset into `num_partitions` with similar class distribution in each partition."""

    # Extract labels from the dataset
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Stratify split using StratifiedKFold
    skf = StratifiedKFold(n_splits=num_partitions, shuffle=True, random_state=seed)

    partitions = []
    for _, test_index in skf.split(np.zeros(len(labels)), labels):
        partitions.append(test_index)

    # Create subsets for each partition
    trainsets = [Subset(dataset, partition) for partition in partitions]

    return trainsets


def prepare_dataset(num_partitions: int = 10, batch_size: int = 5, train_data_ratio: float = 0.8,
                    data_path: str = "./data", target_clt: str = "Client"):
    """Download Data and generate stratified partitions."""

    trainset, testset = get_data(data_path, train_data_ratio)

    # Split trainset into `num_partitions` trainsets (one per client) with stratified distribution
    trainsets = stratified_split(trainset, num_partitions, seed=2023)

    # Print label counts for each partition
    print_label_counts(trainsets, trainset,target_clt)

    # Create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # For each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)  # Total number of images
        num_val = int((1 - train_data_ratio) * num_total)  # Eval portion
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # Construct data loaders and append to their respective list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # We leave the test set intact (i.e., we don't partition it)
    # This test set will be left on the server side and used to evaluate the performance of the global model
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
