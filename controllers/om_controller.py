from modules.agents import REGISTRY as agent_REGISTRY
from modules.am import REGISTRY as am_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from utils.MY_EXP_PATH import EXP_DATA_PATH
import torch as th
import torch.nn as nn
import os
from modules.am.omg_am import RNN


# This multi-agent controller shares parameters between agents, the agent use different algo
class Multi_Module_MAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.main_alg_args = args
        self._build_multi_mudule_config()
        input_shape = self._get_input_shape(scheme, groups)
        self._build_agents(scheme, groups, input_shape)
        
        self.action_selector = action_REGISTRY[self.main_alg_args.action_selector](self.main_alg_args)

        self.hidden_states = None 

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.agent_shift:
            avail_actions = th.cat([avail_actions[:, self.args.agent_shift:, :], avail_actions[:, :self.args.agent_shift, :]], dim=1)
        
        agent_outs = [None] * len(self.algs)
        for i, agent in enumerate(self.agents):
            agent_outs[i], self.hidden_states[i] = agent(agent_inputs[i], self.hidden_states[i])
        agent_outs = th.cat(agent_outs, dim=-2).view(ep_batch.batch_size, -1, self.args.n_actions)
        agent_outs = agent_outs[:, self.agent_alg_idx, :]
        if self.args.agent_shift:
            agent_outs = th.cat([agent_outs[:, -self.args.agent_shift:, :], agent_outs[:, :-self.args.agent_shift, :]], dim=1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def main_alg_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        main_agent_outs = None
        for i, agent in enumerate(self.agents):
            if i != self.main_alg_idx:
                continue
            main_agent_outs, self.hidden_states[i] = agent(agent_inputs[i], self.hidden_states[i])
        return main_agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = []
        for agent in self.agents:
            self.hidden_states.append(agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1))  # bav
        for model in self.agents_model:
            model.init_hidden(batch_size)

    def parameters(self):
        for alg, agent in zip(self.algs, self.agents):
            if alg == self.args.train_alg:
                return agent.parameters()

    def load_state(self, other_mac):
        for i, (alg, agent) in enumerate(zip(self.algs, self.agents)):
            if alg == self.args.train_alg:
                agent.load_state_dict(other_mac.agents[i].state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()
        for am in self.agents_model:
            am.cuda()

    def save_models(self, path):
        for alg, agent, am in zip(self.algs, self.agents, self.agents_model):
            os.makedirs("{}/{}".format(path, alg), exist_ok=True)
            th.save(agent.state_dict(), "{}/{}/agent.th".format(path, alg))
            am.save_models(os.path.join(path, alg))

    def load_models(self, path):
        for alg, agent, am in zip(self.algs, self.agents, self.agents_model):
            if os.path.exists("{}/{}/agent.th".format(path, alg)):
                agent.load_state_dict(th.load("{}/{}/agent.th".format(path, alg), map_location=lambda storage, loc: storage))
            am.load_models(os.path.join(path, alg))

    def set_agents_permutation(self, shift=0):
        shift = shift % self.n_agents
        temp_agent_algs = self.agent_algs[shift:] + self.agent_algs[:shift]
        self.agent_alg_idx = [self.algs.index(a)*self.n_agents + i for i, a in enumerate(temp_agent_algs)]

    def _build_agents(self, scheme, groups, input_shape):
        self.agents = []
        self.agents_model = []
        for i, alg_args in enumerate(self.algs_args):
            self.agents.append(agent_REGISTRY[alg_args.policy_model](input_shape[i][0]+input_shape[i][1][1], self.algs_args[i]))
            self.agents_model.append(am_REGISTRY[alg_args.am_model](scheme, groups, self.algs_args[i]))
                    
    def _build_multi_mudule_config(self):

        import re, copy
        self.algs = re.split(r"[\u0030-\u0039\s]+", self.args.name)[1:]
        nums = list(map(int, re.findall('\d+', self.args.name)))
        if not self.algs:
            self.algs = [self.args.name]
            nums = [self.args.n_agents]
        self.agent_algs = [item for a, n in zip(self.algs, nums) for item in [a]*n]
        self.set_agents_permutation()

        algs_args = []
        for i, alg in enumerate(self.algs):
            temp_args = copy.copy(self.args)
            alg_config = getattr(temp_args, alg)
            del temp_args.__dict__[alg]
            del temp_args.name
            for k, v in alg_config.items():
                setattr(temp_args, k, v)
            if not hasattr(temp_args, "am_model"):
                setattr(temp_args, "am_model", "none_am")
            if alg == self.args.train_alg :
                self.main_alg = self.args.train_alg
                self.main_alg_idx = i
                self.main_alg_args = temp_args
            algs_args.append(temp_args)
        self.algs_args = algs_args

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size

        if self.args.obs_is_state:
            state = batch["state"]
            if len(state.shape) == 3:
                state = state.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            elif len(state.shape) == 4:
                pass
            else:
                raise IndexError

        obs_inputs = []
        if self.args.obs_is_state:
            obs_inputs.append(state[:, t])
        else:
            obs_inputs.append(batch["obs"][:, t])  # b1av
            if self.args.obs_last_action:
                if t == 0:
                    obs_inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    obs_inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            obs_inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        
        inputs = []
        for model in self.agents_model:
            temp = [t.clone() for t in obs_inputs]
            
            temp.append(model.forward(batch, t).squeeze(0).detach())
            temp = th.cat([x.reshape(bs, self.n_agents, -1) for x in temp], dim=-1)
            if self.args.agent_shift:
                temp = th.cat([temp[:, self.args.agent_shift:, :], temp[:, :self.args.agent_shift, :]], dim=1)
            inputs.append(temp.view(bs * self.n_agents, -1))
        return inputs

    def _get_input_shape(self, scheme, groups):
        input_shape = []
        if self.args.obs_is_state:
            temp_shape = scheme["state"]["vshape"]
        else:
            temp_shape = scheme["obs"]["vshape"]
            if self.args.obs_last_action:
                temp_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            temp_shape += self.n_agents
        for alg_args in self.algs_args:
            am_shapes = am_REGISTRY[alg_args.am_model].get_shapes(scheme, groups, self.args)
            input_shape.append([temp_shape, am_shapes])
        return input_shape

    def parse_multi_module(self):
        self.args.name = "6iql1iql_omg"

    def set_agent_shift(self):
        self.args.agent_shift = (self.args.agent_shift + 1) % self.args.n_agents

    def update_agents_parameters(self, alg, files):
        assert self.algs[-1] == alg, "alg error"
        for file in files:
            if file.endswith("agent.th"):
                self.agents[-1].load_state_dict(th.load(file, map_location=lambda storage, loc: storage))
            elif file.endswith("base_am.th"):
                self.agents_model[-1].load_state_dict(th.load(file, map_location=lambda storage, loc: storage))
            elif file.endswith("cvae.th"):
                self.agents_model[-1].cvae.load_state_dict(th.load(file, map_location=lambda storage, loc: storage))
                self.agents_model[-1].fc = nn.Linear(self.agents_model[-1].fc.in_features, self.agents_model[-1].fc.out_features)
                self.agents_model[-1].cond_rnn = RNN(self.agents_model[-1].cond_rnn.f.input_size, self.agents_model[-1].cond_rnn.f.hidden_size)
                cur_device = list(self.agents_model[-1].cvae.parameters())[0].device
                self.agents_model[-1].fc.to(cur_device)
                self.agents_model[-1].cond_rnn.to(cur_device)
            elif file.endswith("vae.th"):
                self.agents_model[-1].vae.load_state_dict(th.load(file, map_location=lambda storage, loc: storage))

    def update_opponents_parameters(self, algs, files):
        for alg in algs:
            idx = self.algs[:-1].index(alg)
            for file in files:
                if file.endswith("agent.th"):
                    self.agents[idx].load_state_dict(th.load(file, map_location=lambda storage, loc: storage))
                else:
                    pass