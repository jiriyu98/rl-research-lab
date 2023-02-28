from collections import deque
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from envs.EmptyRooms import EmptyRooms

import gym

# env
env = EmptyRooms(render_mode="human")
env.reset(seed=0)
torch.manual_seed(0)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
