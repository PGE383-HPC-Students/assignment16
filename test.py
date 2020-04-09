#!/usr/bin/env python
import os
import unittest
from mpi4py import MPI
import numpy as np

from assignment16 import GaussTable

class TestGaussTable(unittest.TestCase):

    def setUp(self):

        comm = MPI.COMM_WORLD
        self.gt = GaussTable(lambda x: x ** 2. * np.exp(-x ** 2.), (0, 1), comm)

    def test_generate_table_5(self):

        table = self.gt.generate_table(5)

        if self.gt.rank == 0:
            np.testing.assert_allclose(np.array(table).T[1], np.array([0.19470019576785122, 0.188321172541532, 0.18953876630452318, 0.18947041612683269, 0.18947238269356842]), atol=1e-6)

    def test_generate_table_10(self):

        table = self.gt.generate_table(10)

        if self.gt.rank == 0:
            np.testing.assert_allclose(np.array(table).T[1], np.array([0.19470019576785122, 0.188321172541532, 0.18953876630452318, 0.18947041612683269, 0.18947238269356842, 0.18947234529808807,  0.1894723458263567, 0.18947234582043798, 0.18947234582049277, 0.18947234582049235]), atol=1e-6)

if __name__ == '__main__':
    unittest.main()
