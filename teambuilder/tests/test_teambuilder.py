import random
import pandas as pd
from teambuilder import TeamBuilder


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
    return pd.DataFrame(dict(name=[f'person-{i}' for i in range(n)],
                        **{c: [random.randint(0, 1) for _ in range(n)] for c in categorical},
                        **{v: [random.randint(45, 90) for _ in range(n)] for v in continuous}))


class TestTeamBuilder:
    def setup(self):
        self.n = 100
        self.id = 'name'
        self.categorical = ['categorical1', 'categorical2']
        self.continuous = ['continuous1', 'countinuous2']
        self.together = [[f'person-{i}' for i in [0, 1, 2, 3, 4]]]
        self.separate = [[f'person-{i}' for i in [5, 6, 7, 8, 9]]]
        self.data = synthetic(self.n,
                              categorical=self.categorical,
                              continuous=self.continuous)

    def test_synthetic(self):
        assert len(self.data) == self.n
        assert list(self.data.columns) == [
            self.id, *self.categorical, *self.continuous]
        assert all(i in [0, 1]
                   for col in self.categorical for i in self.data[col])
        assert all(
            0 <= i <= 100 for col in self.continuous for i in self.data[col])

    def test_shuffle(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=self.together,
                         separate=self.separate)
        tb.shuffle()

        assert list(self.data[self.id]) != list(tb.data[self.id])
        assert all(any(tb.data[self.id].isin([i])) for i in self.data[self.id])

    def test_init_split(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=self.together,
                         separate=self.separate)

        groups = 4 * [20] + [10, 5, 5]
        assert sum(groups) == len(self.data)

        tb._initial_state(groups=groups)

        assert 'group' in tb.data.columns
        for i, n in enumerate(groups):
            assert len(tb.data[tb.data['group'] == i]) == n

        assert len(tb.data[tb.data['group'] == 4]) == 10

    def test_stats(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=self.together,
                         separate=self.separate)

        groups = 10 * [10]
        tb._initial_state(groups=groups)
        stats = tb.stats()

        assert all(i in stats.columns for i in self.categorical +
                   self.continuous)
        assert len(stats) == len(groups) + 1  # +1 is added for totals

    def test_cost(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=self.together,
                         separate=self.separate)

        groups = 10 * [10]
        tb._initial_state(groups=groups)

        assert isinstance(tb.cost(), float)
        assert tb.cost() > 0

    def test_step(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=self.together,
                         separate=self.separate)

        groups = 5 * [20]
        tb._initial_state(groups=groups)

        tb.step()

    def test_solve1(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=[],
                         separate=[])

        groups = 10 * [10]
        tb._initial_state(groups=groups)
        cost_init = tb.cost()

        tb.solve(groups=groups, n=200)

        assert tb.cost() < cost_init
        
    def test_solve2(self):
        tb = TeamBuilder(data=self.data,
                         id=self.id,
                         categorical=self.categorical,
                         continuous=self.continuous,
                         together=self.together,
                         separate=self.separate)

        groups = 10 * [10]
        tb._initial_state(groups=groups)
        tb.shuffle()
        cost_init = tb.cost()

        tb.solve(groups=groups, n=100)

        # Does cost decrease?
        assert tb.cost() < cost_init
        
        # Are people together as expected?
        group = tb.data[tb.data[self.id] == self.together[0][0]]['group'].values[0]
        assert all(i in tb.data[tb.data['group'] == group][self.id].to_list() for i in self.together[0])
        
        # Are people separate as expected?
        assert max(len(set(tb.data[tb.data['group'] == group][self.id]).intersection(self.separate[0]))
                   for group in self.data['group'].unique()) == 1
