import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class None_AM(nn.Module):
    def __init__(self, scheme, group, args):
        super(None_AM, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        # Set up network layers
        self.fc1 = nn.Linear(1, 1)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        return inputs
    
    def init_hidden(self, batch_size):
        pass

    def save_models(self, path):
        pass
    
    def load_models(self, path):
        pass

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        
        return th.zeros((bs, max_t, self.n_agents, 0), device=batch['obs'].device)

    @staticmethod
    def get_shapes(scheme, groups, args):
        input_shape = 0
        output_shape = 0
        return input_shape, output_shape