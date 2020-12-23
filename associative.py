# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File originally create by Raul Peralta-Lozada.

import numpy as np
import random
import time

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
        self.relation = np.zeros((self.m, self.n), dtype=np.bool)

    def __str__(self):
        relation = np.zeros((self.m, self.n), dtype=np.unicode)
        relation[:] = 'O'
        r, c = np.nonzero(self.relation[:self.m,:self.n])
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
    def relation(self, new_relation: np.ndarray):
        if (isinstance(new_relation, np.ndarray) and
                new_relation.dtype == np.bool and
                new_relation.shape == (self.m, self.n)):
            self._relation = new_relation
        else:
            raise ValueError('Invalid relation assignment.')

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        e = 0.0  # entropy
        v = self.relation.sum(axis=0)  # number of marked cells in the columns
        for vi in v:
            if vi != 0:
                e += np.log2(1.0 / vi)
        e *= (-1.0 / self.n)
        return e

    # @classmethod
    # def from_relation(cls, relation: np.ndarray) -> 'AssociativeMemory':
    #     associative_mem = cls(relation.shape[1], relation.shape[0])
    #     associative_mem.relation = relation
    #     return associative_mem

    @property
    def undefined(self):
        return np.nan


    def is_undefined(self, value):
        return np.all(np.isnan(value))


    def vector_to_relation(self, vector):
        relation = np.zeros((self.m, self.n), np.bool)
        try:
            relation[vector, range(self.n)] = True
        except:
            pass
        return relation


    # Choose a value for feature i.
    def choose(self, i, v):
        min = v
        max = v

        for j in range(v, -1, -1):
            if self.relation[j,i]:
                min = j
            else:
                break

        for j in range(v, self.m):
            if self.relation[j,i]:
                max = j
            else:
                break

        if min == max:
            return v
        else:
            k = round(random.triangular(min, max, v))
            return k


    def abstract(self, r_io) -> None:
        self.relation = self.relation | r_io


    def containment(self, r_io):
        return ~r_io | self.relation

    def mismatches(self, r_io):
        result = ~r_io | self.relation
        return np.count_nonzero(result == False)


    # Reduces a relation to a function
    def lreduce(self, vector):
        v = np.full(self.n, self.undefined)

        for i in range(self.n):
            v[i] = self.choose(i, vector[i])

        return v


    def validate(self, vector):
        if vector.size != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size)

        if vector.max() > self.m or vector.min() < 0:
            raise ValueError('Values in the input vector are invalid.')


    def register(self, vector) -> None:
        # Forces it to be a vector.
        vector = np.ravel(vector)

        self.validate(vector)

        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)


    def recognize(self, vector):

        self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)

        return np.all(r_io == True)


    def recall(self, vector):

        self.validate(vector)
        r_io = self.vector_to_relation(vector)
        buffer = self.containment(r_io)

        if np.all(buffer == True):
            # r_io = self.lreduce(r_io)
            r_io = self.lreduce(vector)
        else:
            r_io = np.full(self.n, self.undefined)

        return r_io
