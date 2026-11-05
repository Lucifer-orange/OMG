import copy
from signal import siginterrupt
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class OMGLearner:
    '''Just for OMGMA-agent training'''
    '''It include Q-net and OMG-CVAE-net, it requires that OMG-VAE-net is pre-trained'''
    def __init__(self, mac, scheme, groups, logger, args):
        
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.params += list(mac.agents_model[mac.main_alg_idx].omg_parameters())

        self.last_target_update_episode = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        batch_size = batch.batch_size // self.args.n_agents
        max_t = batch.max_seq_length
        batch = self._refine_batch(batch)

        rewards = batch["reward"][:, :-1].squeeze(-1)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float().squeeze(-1)
        mask = batch["filled"][:, :-1].float().squeeze(-1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate OMG loss
        omg_am = self.mac.agents_model[self.mac.main_alg_idx]
        eval_net = self.mac.agents[self.mac.main_alg_idx]
        omg_loss = omg_am.omg_loss_func(batch, eval_net, self.args.subgoal_mode)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.main_alg_forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.main_alg_forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_td_error ** 2).sum() / mask.sum()
        loss = q_loss + omg_loss
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("omg_loss", omg_loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.cpu().item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _refine_batch(self, batch):
        n_agents = self.args.n_agents
        max_t = batch.max_seq_length
        new_bs = batch.batch_size // n_agents
        assert n_agents * new_bs == batch.batch_size, "The buffer size should be a multiple of the number of agents"
        new_data = {}

        for key in batch.scheme.keys():
            if key == "agent_idx": continue
            elif "group" not in batch.scheme[key].keys():
                temp = batch[key].unsqueeze(2)
                new_data[key] = th.cat([temp[i * n_agents : (i + 1) * n_agents].transpose(0,2) for i in range(new_bs)], dim=0)
            elif "group" in batch.scheme[key].keys():
                idx = th.arange(n_agents).reshape(n_agents,1,1,1).repeat(1, max_t, 1, batch[key].shape[-1]).to(batch.device)
                new_data[key] = th.cat([th.gather(batch[key][i * n_agents : (i + 1) * n_agents], dim=2, index=idx).transpose(0, 2) for i in range(new_bs)], dim=0)
        new_batch = copy.deepcopy(batch[:new_bs])
        new_batch.data.transition_data = new_data
        return new_batch

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        self.mac.agents_model[self.mac.main_alg_idx].omg_save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)