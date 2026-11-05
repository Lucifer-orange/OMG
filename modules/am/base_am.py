import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Base_AM(nn.Module):
    def __init__(self, scheme, groups, args):
        super(Base_AM, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape, output_shape = self.get_shapes(scheme, groups, args)
        self.output_type = "softmax"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_shape)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_shape = x.shape
        x_new_shape = x_shape[:-1] + th.Size([self.n_agents, self.n_actions])
        x = x.view(x_new_shape)
        x = F.softmax(x, dim=-1).view(x_shape)
        return x

    def loss_func(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        #network outputs
        outputs = self.forward(batch)
        x_shape = outputs.shape
        x_new_shape = x_shape[:-1] + th.Size([self.n_agents, self.n_actions])
        outputs = outputs.view(x_new_shape).view(-1,self.n_actions)
        
        # target
        actions = batch["actions"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1).view(-1)
        
        # agent's action mask
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.repeat(bs, max_t, 1, 1).view(-1)
        
        #terminated_mask
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask = mask * (1 - terminated)
        mask = mask.view(bs, max_t, 1, 1).repeat(1, 1, self.n_agents, self.n_agents).view(-1)

        mask = agent_mask * mask

        loss = F.nll_loss(outputs, actions, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def init_hidden(self, batch_size):
        pass

    def save_models(self, path):
        th.save(self.state_dict(), "{}/base_am.th".format(path))
    
    def load_models(self, path):
        self.load_state_dict(th.load("{}/base_am.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observation
        inputs.append(batch["obs"][:, ts])

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # observation
        input_shape = scheme["obs"]["vshape"]
        return input_shape

    @staticmethod
    def get_shapes(scheme, groups, args):
        input_shape = scheme["obs"]["vshape"]
        output_shape = groups["agents"] * groups["n_actions"]
        return input_shape, output_shape