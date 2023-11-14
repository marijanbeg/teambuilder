import random
import pandas as pd


def synthetic(n, categorical=[], continuous=[]):
    """Synthetic dataset.

    For each element in ``categorical``, either 0 or 1 is generated randomly.
    Similarly, for each element in ``continuous``, a random value between 0 and
    100 is generated.

    Parameters
    ----------
    n: int
        Number of people
    categorical: iterable(str), optional
        Categorical properties, e.g. gender, country, etc. Its values will be
        either 0 or 1. Defaults to [].
    values: iterable(str), optional
        Continuous properties, e.g. age, average_mark, etc. Its values will be
        between 0 and 100. Defaults to [].

    Returns
    -------
    pd.DataFrame
        Sythetic dataset

    """
    return pd.DataFrame(
        dict(
            name=[f"person-{i}" for i in range(n)],
            **{c: [random.randint(0, 1) for _ in range(n)] for c in categorical},
            **{v: [random.randint(45, 90) for _ in range(n)] for v in continuous},
        )
    )
