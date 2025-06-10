import pandas as pd
import numpy as np
import pytest

@pytest.fixture(scope="session")
def tiny_df():
    np.random.seed(0)
    df = pd.DataFrame({
        "TransactionID": range(10),
        "TransactionDT": np.random.randint(60, 3600, 10),
        "TransactionAmt": np.random.rand(10) * 100,
        "card4": np.random.choice(["visa", "mastercard", "discover"], 10),
        "isFraud": np.random.randint(0, 2, 10),
    })
    return df
