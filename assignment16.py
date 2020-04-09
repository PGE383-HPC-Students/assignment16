#!/usr/bin/env python
import numpy as np
import scipy.integrate

from mpi4py import MPI

class GaussTable():

    def __init__(self, function, integration_limits, comm):

        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

        self.function = function
        self.integration_limits = integration_limits

        self.my_table = []

    def generate_table(self, N):

        #####################
        ### Add code here ###
        #####################

        return self.clean_table(self.comm.gather(self.my_table, root=0))

    def clean_table(self, table):
        """ Removes empty table entries ([]) and sorts results """
        clean_table = []
        if self.rank == 0:
            for item in table:
                if item != []:
                    for item in item:
                        clean_table += [item]

        return sorted(clean_table) 


if __name__ == "__main__":

    comm  = MPI.COMM_WORLD

    gi = GaussTable(lambda x: x ** 2. * np.exp(-x ** 2.), (0, 1), comm)

    table = gi.generate_table(10)

    if comm.rank == 0:
        print(table)
