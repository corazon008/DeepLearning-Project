import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


if __name__ == '__main__':
    df = pd.read_csv('../data/health_lifestyle_dataset.csv')

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe(include='all'))

    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())

    # drop unnecessary columns
    df = df.drop(columns=['id'])

    # drop duplicate rows
    df = df.drop_duplicates()
    print(f"\nDataset shape after dropping duplicates: {df.shape}")

    # Standardize values
    features_to_scale = [
        'age', 'bmi', 'daily_steps', 'sleep_hours',
        'water_intake_l', 'resting_hr', 'systolic_bp', 'diastolic_bp'
    ]

    regression_targets = ['cholesterol', 'calories_consumed']

    # Standardisation des features
    scaler_x = StandardScaler()
    df[features_to_scale] = scaler_x.fit_transform(df[features_to_scale])

    # Standardisation des targets de la régression
    scaler_y = StandardScaler()
    df[regression_targets] = scaler_y.fit_transform(df[regression_targets])

    # Encode categorical variables
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Matrice de corrélation')
    plt.show()

    # Classification target distribution
    sns.countplot(x='disease_risk', data=df)
    plt.title('Distribution de la cible disease_risk')
    plt.show()

    df.to_csv("../data/health_lifestyle_dataset_cleaned.csv", index=False)