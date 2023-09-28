import random
import pandas as pd
from teambuilder import TeamBuilder


class TestTeamBuilder:
    def setup(self):
        self.data = pd.DataFrame(
            {
                "name": list("abcd"),
                "ctg1": [False, False, True, True],
                #   'ctg2': [True, True, False, False],
                "cnt1": [40, 0, 60, 100],
                #   'cnt2': [100, 100, 0, 0],
            }
        )
        self.identifier = "name"
        self.categorical = ["ctg1"]
        self.continuous = ["cnt1"]
        self.together = [["a", "b"]]
        self.separate = [["a", "b"]]
        self.groups = [2, 2]
        self.group_names = ["group1", "group2"]

        self.tb = TeamBuilder(
            data=self.data,
            identifier=self.identifier,
            groups=self.groups,
            group_names=self.group_names,
            categorical=self.categorical,
            continuous=self.continuous,
            together=self.together,
            separate=self.separate,
        )

    def test_solve1(self):
        self.tb.together = []
        self.tb.separate = []

        ca0 = self.tb.group_cost(0)
        cb0 = self.tb.group_cost(1)
        c0 = self.tb.cost

        self.tb.solve(n_iter=10)

        ca1 = self.tb.group_cost(0)
        cb1 = self.tb.group_cost(1)
        c1 = self.tb.cost

        assert c0 == ca0 + cb0
        assert c1 == ca1 + cb1
        assert c1 <= c0

        m1 = set(self.tb.members("group1"))
        m2 = set(self.tb.members("group2"))

        assert m1 & m2 == set()
        assert m1 | m2 == set("abcd")
        assert m1 == set("ac") or m1 == set("bd")
        assert m2 == set("ac") or m2 == set("bd")

    # def test_solve2(self):
    #     self.separate = []

    #     c0 = self.tb.cost

    #     self.tb.solve(n_iter=10)

    #     c1 = self.tb.cost

    #     assert c1 <= c0

    #     m1 = set(self.tb.members('group1'))
    #     m2 = set(self.tb.members('group2'))

    #     assert m1 & m2 == set()
    #     assert m1 | m2 == set('abcd')
    #     assert m1 == set('ab') or m1 == set('cd')
    #     assert m2 == set('ab') or m2 == set('cd')

    # def setup(self):
    #     self.n = 100
    #     self.identifier = 'name'
    #     self.categorical = ['categorical1', 'categorical2']
    #     self.continuous = ['continuous1', 'countinuous2']
    #     self.together = [[f'person-{i}' for i in [0, 1, 2, 3, 4]]]
    #     self.separate = [[f'person-{i}' for i in [5, 6, 7, 8, 9]]]
    #     self.data = synthetic(self.n,
    #                           categorical=self.categorical,
    #                           continuous=self.continuous)

    # def test_synthetic(self):
    #     assert len(self.data) == self.n
    #     assert list(self.data.columns) == [
    #         self.identifier, *self.categorical, *self.continuous]
    #     assert all(i in [0, 1]
    #                for col in self.categorical for i in self.data[col])
    #     assert all(
    #         0 <= i <= 100 for col in self.continuous for i in self.data[col])

    # def test_initial_state(self):
    #     tb = TeamBuilder(data=self.data,
    #                      identifier=self.identifier,
    #                      categorical=self.categorical,
    #                      continuous=self.continuous,
    #                      together=self.together,
    #                      separate=self.separate)

    #     groups = 4 * [20] + [10, 5, 5]
    #     assert sum(groups) == len(self.data)

    #     tb._initial_state(groups=groups)

    #     assert 'group_number' in tb.data.columns
    #     for i, n in enumerate(groups):
    #         assert len(tb[i]) == n

    #     assert len(tb[4]) == 10

    # # def test_overview(self):
    # #     tb = TeamBuilder(data=self.data,
    # #                      identifier=self.identifier,
    # #                      categorical=self.categorical,
    # #                      continuous=self.continuous,
    # #                      together=self.together,
    # #                      separate=self.separate)

    # #     groups = 10 * [10]
    # #     tb._initial_state(groups=groups)
    # #     overview = tb.overview()

    # #     assert all(i in overview.columns for i in self.categorical +
    # #                self.continuous)
    # #     assert len(overview) == len(groups) + 1  # +1 is added for totals

    # def test_cost(self):
    #     tb = TeamBuilder(data=self.data,
    #                      identifier=self.identifier,
    #                      categorical=self.categorical,
    #                      continuous=self.continuous,
    #                      together=self.together,
    #                      separate=self.separate)

    #     groups = 10 * [10]
    #     tb._initial_state(groups=groups)

    #     assert isinstance(tb.cost, float)
    #     assert tb.cost > 0

    # def test_step(self):
    #     tb = TeamBuilder(data=self.data,
    #                      identifier=self.identifier,
    #                      categorical=self.categorical,
    #                      continuous=self.continuous,
    #                      together=self.together,
    #                      separate=self.separate)

    #     groups = 5 * [20]
    #     tb._initial_state(groups=groups)

    #     tb.step()

    # def test_solve1(self):
    #     tb = TeamBuilder(data=self.data,
    #                      identifier=self.identifier,
    #                      categorical=self.categorical,
    #                      continuous=self.continuous,
    #                      together=[],
    #                      separate=[])

    #     groups = 10 * [10]
    #     tb._initial_state(groups=groups)
    #     cost_init = tb.cost

    #     tb.solve(groups=groups, n_iter=200)

    #     assert tb.cost < cost_init

    # def test_solve2(self):
    #     tb = TeamBuilder(data=self.data,
    #                      identifier=self.identifier,
    #                      categorical=self.categorical,
    #                      continuous=self.continuous,
    #                      together=self.together,
    #                      separate=self.separate)

    #     groups = 10 * [10]
    #     tb._initial_state(groups=groups)
    #     cost_init = tb.cost

    #     tb.solve(groups=groups, n_iter=100)

    #     # Does cost decrease?
    #     assert tb.cost < cost_init

    #     # Are people together as expected?
    #     group = tb.data[tb.data[self.identifier] == self.together[0][0]]['group_number'].values[0]
    #     assert all([i in tb.members(group) for i in self.together[0]])

    #     # Are people separate as expected?
    #     assert max(len(set(tb.members(group)).intersection(self.separate[0]))
    #                for group in self.data['group_number'].unique()) == 1
