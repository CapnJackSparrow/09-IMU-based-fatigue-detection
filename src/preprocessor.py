'''Ths script defines a function which scales and normalizes a Pandas DataFrame
containing features for training a Machine Learning model.'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocessing(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """
    Preprocesses a Pandas DataFrame by:
    1. Standardizing the data using StandardScaler.
    2. Performing PCA with the specified number of components.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_components (int, optional): Number of PCA components to retain. Defaults to 2.

    Returns:
        pd.DataFrame: Processed DataFrame with standardized features and PCA components.
    """
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Standardize the data
    scaled_data = scaler.fit_transform(df)

    # Initialize PCA
    pca = PCA(n_components=n_components)

    # Fit PCA to the standardized data
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame with the PCA components
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])

    return pca_df
