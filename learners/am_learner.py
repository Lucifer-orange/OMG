from ast import Name
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.am import REGISTRY as am_REGISTRY
import torch as th
from torch.optim import RMSprop
import os

class AMLearner:
    '''Just for am-model training, use am-model's loss_func to get the loss'''
    def __init__(self, mac, scheme, groups, logger, args):
        
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.am_model = am_REGISTRY[args.am_model](scheme, groups, args)
        self.params = list(self.am_model.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        loss = self.am_model.loss_func(batch)
        
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.cpu().item(), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.am_model.cuda()
        
    def save_models(self, path):
        self.am_model.save_models(path)

    def load_models(self, path):
        if self.args.mac == 'basic_mac':
            self.mac.load_models(os.path.join(path, self.args.name))
        elif self.args.mac == 'multi_module_mac':
            self.mac.load_models(path)
        else:
            raise NameError