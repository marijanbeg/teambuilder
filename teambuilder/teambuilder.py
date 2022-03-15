import random
import functools
import collections.abc
import pandas as pd
import numpy as np

random.seed(10)


class TeamBuilder:
    def __init__(self, data, id, categorical=[], continuous=[],
                 together=[], separate=[]):
        self.data = data
        self.id = id
        self.categorical = categorical
        self.continuous = continuous
        self.together = together
        self.separate = separate

    def shuffle(self, random_state=None):
        """Shuffle ``self.data``"""
        self.data = self.data.sample(
            frac=1, random_state=random_state, ignore_index=True)

    def _initial_state(self, groups):
        """Initial split of data into groups.

        By passing an iterable via ``groups``, the exact number of people per
        group is specified. The sum of values in ``groups`` iterable must be the
        same as the number of rows in ``self.data``. Column ``group`` is added
        to ``self.data``.

        Parameters
        ----------
        groups: Iterable(int)
            The number of people per group.

        """
        if not isinstance(groups, collections.abc.Iterable):
            raise TypeError(f'Unsupported for {type(groups)=}.')
        if sum(groups) != len(self.data):
            raise ValueError(
                f'The sum of chunks must be the same as {len(self.data)=}.')

        for i, (chunk, start) in enumerate(zip(groups, np.cumsum([0, *groups]))):
            self.data.loc[start: start+chunk, 'group'] = i

    def stats(self):
        """Statistics overview.

        Computes the statistics "per group". For categorical variables, it
        computes the number of people with that characteristic. For continuous
        variables, it computes the mean.

        Returns
        -------
        pd.DataFrame
            Statistics overview.

        """
        if 'group' not in self.data.columns:
            raise ValueError(
                'Data must be reduced before the statistics can be shown.')

        per_group = dict(**{i: self.data[self.data[i] == 1].groupby('group').count()[i] for i in self.categorical},
                         **{i: self.data.groupby('group')[i].mean() for i in self.continuous},
                         n=self.data.groupby('group').count()[self.categorical[0]])

        totals = dict(**{i: self.data[self.data[i] == 1].count()[i] for i in self.categorical},
                      **{i: self.data[i].mean() for i in self.continuous},
                      n=len(self.data))

        return pd.concat([pd.DataFrame(per_group), pd.DataFrame(totals, index=['total'])]).fillna(0)

    def cost(self):
        """Cost function.

        Returns
        -------
        float
            Cost

        """
        totals = dict(**{i: self.data[self.data[i] == 1].count()[i] / len(self.data) for i in self.categorical},
                      **{i: self.data[i].mean() for i in self.continuous})

        per_group = dict(**{i: (self.data[self.data[i] == 1].groupby('group').count()[i] /
                                self.data.groupby('group').count()[i]) for i in self.categorical},
                         **{i: self.data.groupby('group')[i].mean() for i in self.continuous})

        per_group = {k: v.fillna(0) for k, v in per_group.items()}

        return sum([abs(per_group[i] - totals[i]).sum() for i in self.categorical] +
                   [0.01*abs(per_group[i] - totals[i]).sum() for i in self.continuous])

    @property
    def _partial(self):
        return functools.partial(self.__class__,
                                 id=self.id,
                                 categorical=self.categorical,
                                 continuous=self.continuous,
                                 together=self.together,
                                 separate=self.separate)

    def step(self, cost=None):
        """Relaxation step.

        Swaps two random people between two different groups. If the cost
        function decreases, the step is accepted.

        Parameters
        ----------
        cost: callable, optional
            Cost function. If ``None``, ``self.cost`` is used. Defaults to
            ``None``.

        """
        # If there is only one group, minimisation does not make sense.
        if self.data.groupby('group').count().shape[0] <= 1:
            raise ValueError('There must be at least two groups in data.')
        
        cost = cost or self.cost

        n = len(self.data)  # total number of students
        original = self._partial(self.data.copy())  # deep copy

        c_init = original.cost()  # original cost

        if 'fixed' not in original.data.columns:
            original.data['fixed'] = 0

        # Select two students from different groups and ensure they are not fixed.
        while True:
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)

            if (original.data.iloc[a]['group'] != original.data.iloc[b]['group'] and
                    not original.data.iloc[a]['fixed'] and not original.data.iloc[b]['fixed']):
                break

        # Swap two random students from different groups.
        (original.data.at[a, 'group'], original.data.at[b, 'group']) = (
            original.data.iloc[b]['group'], original.data.iloc[a]['group'])

        self.data = original.data if original.cost() < c_init else self.data


    def solve(self, groups, n, cost=None):
        """Build teams.
    
        Parameters
        ----------
        groups: Iterable(int)
            The number of people per group.
            
        n: int
            Total number of steps.
        
        cost: callable, optional
            Cost function. If ``None``, ``self.cost`` is used. Defaults to
            ``None``.
        
        """
        if 'group' not in self.data.columns:
            self._initial_state(groups=groups)

        for i in range(n):
            self.step(cost=cost)

            if i % 500 == 0:
                print(f'step = {i: 6d}: cost = {self.cost(): .3f}')
