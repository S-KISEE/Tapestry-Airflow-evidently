import pandas as pd

from sklearn.datasets import make_regression


def create_dummy_dataset(
        n_samples: int=100, 
        n_features: int=4, 
        noise: int=0.1, 
        random_state: int=42,
        save_to_csv: bool=False) -> pd.DataFrame:
    """
    Create a dummy dataset using sklearn make_regression

    Parameters
    -------
    n_samples (int)
        -
    
    n_features (int)
        -

    nose (int)
        -

    random_state (int)
        -

    save_to_csv (bool)
        -

    Return
    ------
    pd.DataFrame - dummy dataset
    """
    # Generate dummy regression data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)

    # Convert to DataFrame
    feature_names = [f'feature{i+1}' for i in range(X.shape[1])]
    training_data = pd.DataFrame(X, columns=feature_names)
    training_data['target'] = y

    # Save the DataFrame to a CSV file
    if save_to_csv:
        training_data.to_csv('/path/to/training_data.csv', index=False)

    return training_data