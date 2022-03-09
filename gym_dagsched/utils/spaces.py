import numpy as np
from gym.spaces import MultiBinary, Discrete, Box

from ..args import args
from .misc import triangle


# time can be any non-negative real number
time_spaces = lambda n: Box(low=0., high=np.inf, shape=(n,))
time_space = time_spaces(1)

# exclusive discrete space 0 <= k < n, with
# additional invalid state := n
# e.g. used to represent id's with n = invalid id
discrete_x = lambda n: Discrete(n+1)

# inclusive discrete space 0 <= k <= n
# e.g. used to represent a count of something
discrete_i = lambda n: Discrete(n+1)

# lower triangle of the dag's adgacency matrix stored 
# as a flattened array
dag_space = MultiBinary(triangle(args.max_stages))

# used to indicate frontier stages and saturated stages
stages_mask_space = MultiBinary(args.n_jobs * args.max_stages)