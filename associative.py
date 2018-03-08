#!/usr/bin/env python

# Created by Raul Peralta-Lozada

import numpy


class AssociativeMemoryError(Exception):
    pass


class AssociativeMemory(object):
    def __init__(self, n: int, m: int):
        """
        Parameters
        ----------
        n : int
            The size of the domain.
        m : int
            The size of the range.
        """
        self.n = n
        self.m = m
        self.grid = numpy.zeros((self.m, self.n), dtype=numpy.bool)

    def __str__(self):
        grid = numpy.zeros(self.grid.shape, dtype=numpy.unicode)
        grid[:] = 'O'
        r, c = numpy.nonzero(self.grid)
        for i in zip(r, c):
            grid[i] = 'X'
        return str(grid)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value: int):
        if value > 0:
            self._n = value
        else:
            raise ValueError('Invalid value for n.')

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value: int):
        if value > 0:
            self._m = value
        else:
            raise ValueError('Invalid value for m.')

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, new_grid: numpy.ndarray):
        if (isinstance(new_grid, numpy.ndarray) and
                new_grid.dtype == numpy.bool and
                new_grid.shape == (self.m, self.n)):
            self._grid = new_grid
        else:
            raise ValueError('Invalid grid assignment.')

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        e = 0.0  # entropy
        v = self.grid.sum(axis=0)  # number of marked cells in the columns
        for vi in v:
            if vi != 0:
                e += numpy.log2(1. / vi)
        e *= (-1.0 / self.n)
        return e

    @classmethod
    def from_grid(cls, grid: numpy.ndarray) -> 'AssociativeMemory':
        associative_mem = cls(grid.shape[1], grid.shape[0])
        associative_mem.grid = grid
        return associative_mem

    @staticmethod
    def vector_to_grid(vector, input_range, min_value):
        # now is only binary
        vector = numpy.ravel(vector)
        n = vector.size
        if vector.max() > input_range or vector.min() < min_value:
            raise ValueError('Values in the input vector are invalid.')
        grid = numpy.zeros((input_range, n), numpy.bool)
        vector -= min_value
        grid[vector, range(vector.shape[0])] = True
        grid = numpy.flipud(grid)
        return grid

    def abstract(self, vector_input, input_range=2, min_value=0) -> None:
        if vector_input.size != self.n:
            raise ValueError('Invalid size of the input data.')
        else:
            grid_input = self.vector_to_grid(vector_input, input_range,
                                             min_value)
            self.grid = self.grid | grid_input

    def reduce(self, vector_input, input_range=2, min_value=0):
        if vector_input.size != self.n:
            raise AssociativeMemoryError('Invalid size of the input data.')
        else:
            grid_input = self.vector_to_grid(vector_input,
                                             input_range, min_value)
            grid_output = numpy.zeros(self.grid.shape, dtype=self.grid.dtype)
            for i, cols in enumerate(zip(self.grid.T, grid_input.T)):
                (i1, ) = numpy.nonzero(cols[0])
                (i2, ) = numpy.nonzero(cols[1])
                if numpy.all(numpy.in1d(i2, i1)):
                    # TODO: finish the reduce operation
                    if i1.size == i2.size:
                        pass
                        # grid_output[0:255, i] =
                else:
                    raise AssociativeMemoryError('Applicability '
                                                 'condition error.')
            return grid_input
