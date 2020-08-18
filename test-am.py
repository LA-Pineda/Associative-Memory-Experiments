import numpy as np
from associative import *

# Memory for 4 features of size 3
m = AssociativeMemory(4,3)

# 0: 1 1 1 1
# 1: 0 0 0 0
# 2: 0 0 0 0
v0 = np.array([0, 0, 0, 0])
v1 = np.array([1, 1, 1, 1])
v2 = np.array([2, 2, 2, 2])

# 0: 1 0 0 0
# 1: 0 1 0 1
# 2: 0 0 1 0
vd = np.array([0, 1, 2, 1])

# 0: 0 0 0 1
# 1: 1 0 1 0
# 2: 0 1 0 0

vi = np.array([1, 2, 1, 0])
