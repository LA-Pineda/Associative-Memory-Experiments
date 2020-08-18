# Created by Raul Peralta-Lozada

import numpy
import random


class AssociativeMemoryError(Exception):
    pass


class AssociativeMemory(object):
    def __init__(self, n: int, m: int):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        """
        self.n = n
        self.m = m

        # it is m+1 to handle partial functions.
        self.relation = numpy.zeros((self.m+1, self.n), dtype=numpy.bool)

    def __str__(self):
        relation = numpy.zeros((self.m, self.n), dtype=numpy.unicode)
        relation[:] = 'O'
        r, c = numpy.nonzero(self.relation[:self.m,:self.n])
        for i in zip(r, c):
            relation[i] = 'X'
        return str(relation)

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
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, new_relation: numpy.ndarray):
        if (isinstance(new_relation, numpy.ndarray) and
                new_relation.dtype == numpy.bool and
                new_relation.shape == (self.m+1, self.n)):
            self._relation = new_relation
        else:
            raise ValueError('Invalid relation assignment.')

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        e = 0.0  # entropy
        v = self.relation[:self.m,:self.n].sum(axis=0)  # number of marked cells in the columns
        for vi in v:
            if vi != 0:
                e += numpy.log2(1. / vi)
        e *= (-1.0 / self.n)
        return e

    # @classmethod
    # def from_relation(cls, relation: numpy.ndarray) -> 'AssociativeMemory':
    #     associative_mem = cls(relation.shape[1], relation.shape[0])
    #     associative_mem.relation = relation
    #     return associative_mem


    @property
    def undefined(self):
        return self.m

    
    def is_defined(self, v):
        return self.m == v


    def vector_to_relation(self, vector):
        relation = numpy.zeros((self.m+1, self.n), numpy.bool)
        relation[vector, range(self.n)] = True
        return relation


    # Choose a value for feature i.
    def choose(self, i):
        candidates = []
        
        for j in range(self.m):
            if self.relation[j, i]:
                candidates.append(j)
        
        if len(candidates) != 0:
            k = random.randrange(len(candidates))
            return candidates[k]
        else:
            return self.undefined
        

    def abstract(self, r_io) -> None:
        self.relation = self.relation | r_io


    def containment(self, r_io):
        return ~r_io | self.relation


    # Reduces a relation to a function
    def lreduce(self, relation):
        v = numpy.full(self.n, self.undefined, numpy.int16)

        for i in range(self.n):
            v[i] = self.choose(i)
        
        return v

    
    def wrap(self, relation):
        wrapper = numpy.zeros((self.m+1, self.n), dtype=numpy.bool)

        for i in range(self.m):
            for j in range(self.n):
                print(i)
                

    def validate(self, vector):
        if vector.size != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size)

        if vector.max() > self.m or vector.min() < 0:
            raise ValueError('Values in the input vector are invalid.')


    def register(self, vector) -> None:
        # Forces it to be a vector.
        vector = numpy.ravel(vector)

        self.validate(vector)

        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)


    def recognize(self, vector):

        self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        
        return numpy.all(r_io[:self.m,:self.n] == True)


    def recall(self, vector):

        self.validate(vector)
        r_io = self.vector_to_relation(vector)
        buffer = self.containment(r_io)

        if numpy.all(r_io[:self.m,:self.n] == True):
            r_io = self.lreduce(r_io)
        else:
            r_io = numpy.zeros((self.m+1, self.n), dtype=numpy.bool)

        return r_io