import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(data):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='claim_status', data=data)
    plt.title('Distribution of Claim Status')
    plt.xlabel('Claim Status')
    plt.ylabel('Count')
    plt.show()

    numerical_columns = ['subscription_length', 'vehicle_age', 'customer_age']
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()

    categorical_columns = ['region_code', 'segment', 'fuel_type']
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(3, 1, i)
        sns.countplot(y=column, data=data, order=data[column].value_counts().index)
        plt.title(f'Distribution of {column}')
        plt.xlabel('Count')
        plt.ylabel(column)
    plt.tight_layout()
    plt.show()
