import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from functools import reduce

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(RNN, self).__init__()
        self.f = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, inputs, hidden_state):
        x_shape = inputs.shape
        x = inputs.reshape(-1, x_shape[-1])
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h_out = self.f(x, h_in)
        #raise NotImplementedError
        return h_out.view(x_shape[:-1] + th.Size([-1])), h_out.detach()

    def init_hidden(self):
        return self.f.weight_hh.new(1, self.hidden_dim).zero_()
    
class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def _encode(self, input):
        raise NotImplementedError

    def _decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size, current_device, **kwargs):
        raise NotImplementedError

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass

class VanillaVAE(BaseVAE):
    def __init__(self, args, input_dim, latent_dim, hidden_dims, **kwargs):
        super(VanillaVAE, self).__init__()

        self.args = args
        self.latent_dim = latent_dim

        modules = []

        # Build Encoder
        i_d = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(i_d, h_dim),
                    nn.LeakyReLU())
            )
            i_d = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        
        self.decoder_output = nn.Linear(hidden_dims[-1], input_dim)
    
    def forward(self, inputs):
        mu, log_var = self._encode(inputs)
        z = self._reparameterize(mu, log_var)
        return  z

    def loss_func(self, inputs, mask):
        mu, log_var = self._encode(inputs)
        z = self._reparameterize(mu, log_var)
        recons = self._decode(z)
        recon_loss = F.mse_loss(inputs * mask, recons * mask)
        kld_loss = th.mean(-0.5 * mask * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1, keepdim=True))
        return recon_loss, kld_loss

    def generate(self, inputs):
        mu, log_var = self._encode(inputs)
        z = self._reparameterize(mu, log_var)
        recons = self._decode(z)
        return [inputs, recons, mu, log_var, z]

    def _encode(self, inputs):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(inputs)
        result = th.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def _decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.decoder_output(result)
        return result

    def _reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return eps * std + mu

class ConditionalVAE(BaseVAE):
    def __init__(
            self,
            args, 
            input_dim,  #s+a
            conditioned_input_dim,  #s+a+a_o
            latent_dim,  #gauss disturbution
            hidden_dims,
            **kwargs):
        super(ConditionalVAE, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.condition_input_dim = conditioned_input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        modules = []

        # Build Encoder
        i_d = input_dim + conditioned_input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(i_d, h_dim),
                                         nn.LeakyReLU()))
            i_d = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim + conditioned_input_dim,
                                       hidden_dims[-1])
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                                         nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.decoder_output = nn.Linear(hidden_dims[-1], input_dim)

    def forward(self, inputs, cond_inputs):
        x = th.cat([inputs, cond_inputs], dim=-1)
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        return z

    def forward_full(self, inputs, cond_inputs):
        x = th.cat([inputs, cond_inputs], dim=-1)
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        return z, mu, log_var

    def _encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = th.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def _decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.decoder_output(result)
        return result

    def _reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return eps * std + mu

    def generate(self, inputs, cond_inputs):
        x = th.cat([inputs, cond_inputs], dim=-1)
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        
        x = th.cat([z, cond_inputs], dim=-1)
        recons = self._decode(x)
        return [inputs, recons, mu, log_var, z]

class OMG_AM(nn.Module):
    def __init__(self, scheme, groups, args) -> None:
        super(OMG_AM, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        fc_input, cond_input, vae_input = self._get_input_shape(scheme, groups)
        self._hidden_dim = 128
        self._subgoal_dim = 64
        self.output_type = "embedding"

        # Set up network layers
        self.fc = nn.Linear(fc_input, self._hidden_dim // 4)
        self.cond_rnn = RNN(cond_input, self._hidden_dim // 4 * 3)
        self.cvae = ConditionalVAE(args=args, input_dim=self._hidden_dim // 4, conditioned_input_dim=self._hidden_dim // 4 * 3, latent_dim=self._subgoal_dim, hidden_dims=[128, 128])
        self.vae = VanillaVAE(args=args, input_dim=vae_input, latent_dim=self._subgoal_dim, hidden_dims=[128, 128])
    
    # execute CVAE
    def forward(self, batch, t=None):
        """
            used by mac.bulid_input
            @input: batch
            @output: cvae_embedding(batch_size, traj_len, n_agents, embed_dim)
        """
        fc_inputs, cond_inputs = self._build_inputs(batch, t)
        old_shape = fc_inputs.shape[:-1]
        num = reduce((lambda x, y: x * y), list(fc_inputs.shape[:-1]))
        x = F.relu(self.fc(fc_inputs)).view(num, -1)
        cond_x, self.hidden_state = self.cond_rnn(cond_inputs, self.hidden_state)
        out, mu, log_var = self.cvae.forward_full(x, cond_x.view(num, -1))
        batch.holdup({"infer_mu":mu, "infer_log_var":log_var})
        out = out.view(old_shape + th.Size([-1]))
        return out
    
    # train CVAE
    def omg_loss_func(self, batch, eval_net, subgoal_mode, agent_idx=None):
        '''
            used by omg_learner
            @input: batch, eval_net
            @output: cvae.recon_loss + omg.kl_loss
        '''
        bs = batch.batch_size
        max_t = batch.max_seq_length
        agent_mask = None


        terminated = batch["terminated"][:, :-1].float().squeeze(-1)
        infer_mu = batch["infer_mu"][:, :-1].float().squeeze(-1)
        infer_log_var = batch["infer_log_var"][:, :-1].float().squeeze(-1)
        mask = batch["filled"][:, :-1].float().squeeze(-1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        cvae_input = []
        cvae_output = []
        cvae_mu = []
        cvae_log_var = []

        self.init_hidden(batch.batch_size)
        for t in range(max_t - 1):
            fc_inputs, cond_inputs = self._build_inputs(batch, t)
            x = F.relu(self.fc(fc_inputs))
            cond_x, self.hidden_state = self.cond_rnn(cond_inputs, self.hidden_state)
            x, recons, mu, log_var, _ = self.cvae.generate(x.view(bs * self.n_agents, -1), cond_x.view(bs * self.n_agents, -1))
            
            cvae_input.append(x.view(bs, 1, self.n_agents, -1))
            cvae_output.append(recons.view(bs, 1, self.n_agents, -1))
            cvae_mu.append(mu.view(bs, 1, self.n_agents, -1))
            cvae_log_var.append(log_var.view(bs, 1, self.n_agents, -1))
        
        cvae_input = th.cat(cvae_input, dim=1)
        cvae_output = th.cat(cvae_output, dim=1)
        cvae_mu = th.cat(cvae_mu, dim=1)
        cvae_log_var = th.cat(cvae_log_var, dim=1)

        #Calculate VAE subgoal & mu & log_var
        subgoal, vae_mu, vae_log_var = self._cal_subgoal(batch, eval_net, subgoal_mode, agent_mask)
        
        recons_loss = F.mse_loss(cvae_output, cvae_input, reduction='none').mean(dim=-1)
        recons_loss = recons_loss * mask
        
        if self.args.eta < np.random.random():
            omg_loss = -0.5 * th.sum(1 + cvae_log_var - vae_log_var.detach() - ((vae_mu.detach()-cvae_mu) ** 2 + cvae_log_var.exp())/vae_log_var.detach().exp(), dim=-1)
        else:
            omg_loss = -0.5 * th.sum(1 + cvae_log_var - infer_log_var.detach() - ((infer_mu.detach()-cvae_mu) ** 2 + cvae_log_var.exp())/infer_log_var.detach().exp(), dim=-1)
        omg_loss = omg_loss * mask

        loss = recons_loss.mean() + self.args.omg_cvae_alpha * omg_loss.mean()
        return loss

    def _cal_subgoal(self, batch, eval_net, subgoal_mode, agent_mask):
        #Calculate subgoal & mu & log_var by VAE
        bs = batch.batch_size
        max_t = batch.max_seq_length
        horizon = self.args.omg_horizon
        with th.no_grad():
            obs_for_eval = self._build_obs(batch)[:, :-1]
            if self.args.obs_is_state:
                obs_for_vae = batch["state"][:, 1:]
            else:
                obs_for_vae = batch["obs"][:, 1:]
            obs_for_vae = obs_for_vae.reshape(-1, obs_for_vae.shape[-1])
            
            _, _, mu, log_var, z = self.vae.generate(obs_for_vae)
            subgoal = z.view(bs, max_t-1, self.n_agents, -1)
            mu = mu.view(bs, max_t-1, self.n_agents, -1)
            log_var = log_var.view(bs, max_t-1, self.n_agents, -1)

            #Prepare eval_net inputs
            obs_for_eval = obs_for_eval.transpose(1,2).unsqueeze(3).repeat(1,1,1,horizon,1)
            temp_subgoal = subgoal.transpose(1,2)
            subgoal_for_eval = [temp_subgoal.unsqueeze(3)]
            for h in range(1, horizon):
                temp = th.cat([temp_subgoal[:,:,h:,:], temp_subgoal[:,:,-1:,:].repeat(1,1,h,1)], dim=2).unsqueeze(3)
                subgoal_for_eval.append(temp)
            subgoal_for_eval = th.cat(subgoal_for_eval, dim=3)

            eval_inputs = th.cat([obs_for_eval, subgoal_for_eval],dim=-1)
            eval_inputs = eval_inputs.transpose(1,2)
            eval_inputs = eval_inputs.transpose(2,3)
            
            #Calculate value by eval_net
            value = []
            hidden_state = eval_net.init_hidden().unsqueeze(0).expand(bs, horizon, self.n_agents, -1)
            for t in range(batch.max_seq_length - 1):
                q, hidden_state = eval_net(eval_inputs[:,t].reshape(-1, eval_inputs.shape[-1]), hidden_state)
                value.append(q.view(bs, horizon, self.n_agents, -1).unsqueeze(1).mean(dim=-1))
                hidden_state = hidden_state.view(bs, horizon, self.n_agents, -1)
                hidden_state = hidden_state[:,0:1, :, :].repeat(1, horizon, 1, 1)
            value = th.cat(value, dim=1)

            #obtain k by argmax/argmin V(o_t, subgoal_{t+k})
            if subgoal_mode:
                k = th.argmax(value, dim=2)
            else:
                k = th.argmin(value, dim=2)
            base_idx = th.arange(max_t-1).expand(bs * self.n_agents, -1).view(bs, self.n_agents, -1).transpose(1,2).to(k.device)
            k = th.min(base_idx + k, th.LongTensor([max_t-1]).to(k.device)).unsqueeze(-1).repeat(1,1,1,64)
            
            #Get results
            final_subgoal = th.gather(subgoal, 1, k)
            final_mu = th.gather(mu, 1, k)
            final_log_var = th.gather(log_var, 1, k)
        return final_subgoal, final_mu, final_log_var

    def _build_obs(self, batch):
        bs = batch.batch_size
        max_t = batch.max_seq_length

        inputs = []
        if self.args.obs_is_state:
            state = batch["state"]
            if len(state.shape) == 3:
                state = state.unsqueeze(2).repeat(1,1,self.n_agents,1)
            elif len(state.shape) == 4:
                pass
            else:
                raise IndexError
            inputs.append(state)
        else:
            inputs.append(batch["obs"])  # b1av
            if self.args.obs_last_action:
                inputs.append(th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs * max_t, -1, -1).view(bs, max_t, self.n_agents, self.n_agents))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def loss_func(self, batch):
        '''
            used by am_learner
            @input: batch
            @output: vae_loss
        '''
        inputs, mask = self._build_vae_inputs(batch)
        recons_loss, kld_loss = self.vae.loss_func(inputs, mask)
        return recons_loss + self.args.omg_vae_alpha * kld_loss
    
    def _build_vae_inputs(self, batch):
        if self.args.obs_is_state:
            obs = batch["state"][:, :-1]
        else:
            obs = batch["obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        return obs.reshape(-1, obs.shape[-1]), mask.unsqueeze(2).repeat(1,1,self.n_agents,1).reshape(-1, 1)

    def parameters(self):
        return self.vae.parameters()
    
    def save_models(self, path):
        th.save(self.vae.state_dict(), "{}/vae.th".format(path))

    def load_models(self, path):
        self.vae.load_state_dict(th.load("{}/vae.th".format(path), map_location=lambda storage, loc: storage))
 
    def omg_parameters(self):
        return self.cvae.parameters()

    def omg_save_models(self, path):
        th.save(self.cvae.state_dict(), "{}/cvae.th".format(path))
        th.save(self.cond_rnn.state_dict(), "{}/cond_rnn.th".format(path))
        th.save(self.fc.state_dict(), "{}/fc.th".format(path))
    
    def omg_load_models(self, path):
        self.cvae.load_state_dict(th.load("{}/cvae.th".format(path), map_location=lambda storage, loc: storage))
        self.cond_rnn.load_state_dict(th.load("{}/cond_rnn.th".format(path), map_location=lambda storage, loc: storage))
        self.fc.load_state_dict(th.load("{}/fc.th".format(path), map_location=lambda storage, loc: storage))

    def init_hidden(self, batch_size):
        self.hidden_state = self.cond_rnn.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        fc_inputs = []
        
        # Unified state's shape
        if self.args.obs_is_state:
            state = batch["state"]
            if len(state.shape) == 3:
                state = state.unsqueeze(2).repeat(1,1,self.n_agents,1)
            elif len(state.shape) == 4:
                pass
            else:
                raise IndexError

        # observation
        if self.args.obs_is_state:
            fc_inputs.append(state[:, ts])
        else:
            fc_inputs.append(batch["obs"][:, ts])
        
        cond_inputs = []
        if self.args.obs_is_state:
            last_obs = th.cat([th.zeros_like(state[:, 0:1]), state[:, :-1]], dim=1)
            if t == 0:
                cond_inputs.append(th.zeros_like(state[:, slice(0, 1)]))
            else:
                cond_inputs.append(last_obs[:, ts])
        else:
            last_obs = th.cat([th.zeros_like(batch["obs"][:, 0:1]), batch["obs"][:, :-1]], dim=1)
            if t == 0:
                cond_inputs.append(th.zeros_like(batch["obs"][:, slice(0, 1)]))
            else:
                cond_inputs.append(last_obs[:, ts])

        # other agents last actions
        last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
        if t == 0:
            cond_inputs.append(th.zeros_like(batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)))
        else:
            actions = last_actions[:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
            agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
            cond_inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        
        # self last actions
        if t == 0:
            cond_inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
        else:
            cond_inputs.append(last_actions[:, ts])

        fc_inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in fc_inputs], dim=-1)
        cond_inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in cond_inputs], dim=-1)
        return fc_inputs, cond_inputs

    def _get_input_shape(self, scheme, groups):
        if self.args.obs_is_state:
            obs_shape = scheme["state"]["vshape"]
        else:
            obs_shape = scheme["obs"]["vshape"]

        fc_input = obs_shape
        if self.args.omg_pre_action:
            fc_input += scheme["actions_onehot"]["vshape"][0]
            raise NotImplementedError
        cond_input = obs_shape + (self.n_agents + 1) * self.n_actions
        vae_input = obs_shape

        return fc_input, cond_input, vae_input

    @staticmethod
    def get_shapes(scheme, groups, args):
        if args.obs_is_state:
            input_shape = scheme["state"]["vshape"]
        else:
            input_shape = scheme["obs"]["vshape"]
        output_shape = 64
        return input_shape, output_shape