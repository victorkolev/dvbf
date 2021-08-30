import torch
import torch.nn as nn
import math


def get_transition_model(transition_model, **kwargs):
    """Returns TransitionModel object."""
    transition_model = transition_model.split("_")
    transition_model = [s.capitalize() for s in transition_model]
    transition_model = "".join(transition_model)
    return eval("{}TransitionModel".format(transition_model))


class BayesFilter(nn.Module):
    def __init__(
        self,
        transition_model,
        noise_dim,
        action_dim,
        latent_dim,
        input_dim,
        hidden_size,
        kl_weight,
        annealing_steps=100,
    ):
        r"""
        Deep Variational Bayes Filter as described in [1]

        Parameters
        --------
        transition_model : TransitionModel object
            Parametrizes transition f(z_t, u_t, β_t)

        latent_dim : dim of z_t
        input_dim : dim of x_t
        action_dim : dim of u_t
        noise_dim : dim of w_t

        hidden_size : universal hidden size for networks
        kl_weight : weight of KL divergence term in VAE loss
        annealing_steps : number of steps before annealing rate becomes 1

        References:
            [1] Karl, Maximilian, et al. "Deep variational bayes filters: Unsupervised
            learning of state space models from raw data." arXiv preprint arXiv:1605.06432 (2016).

        """
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.kl_weight = kl_weight

        self.transition_model = transition_model

        # nn.Sequential MLP w/ relu
        # input is self.x /observations/
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.inference = nn.Sequential(
            nn.Linear(noise_dim + latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * noise_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

        self.initial_net = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            dropout=0.25,
        )
        self.initial_affine = nn.Linear(2 * hidden_size, 2 * noise_dim)

        self.initial_noise_to_latent = nn.Sequential(
            nn.Linear(noise_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.anneal_steps = annealing_steps
        if self.anneal_steps != 0:
            self.anneal_rate = 1e-3
            self.update_annealing = self._update_annealing

        else:
            self.anneal_rate = 1.
            self.update_annealing = lambda *args: None

    def _update_annealing(self):
        if self.anneal_rate > 1.0 - 1e-5:
            self.anneal_rate = 1.0
        else:
            self.anneal_rate += 1.0 / self.anneal_steps

    def forward(self, observations, actions, logger=None):
        seq_len, batch_size = observations.shape[:2]

        transition_parameters = self.extractor(
            observations.view(-1, observations.shape[2:])
        ).view(seq_len, batch_size, -1)

        initial_w, _ = self.initial_net(transition_parameters)
        initial_w = initial_w.view(seq_len, batch_size, 2, -1)
        initial_w = torch.cat([initial_w[-1, :, 0], initial_w[0, :, 1]], dim=-1)
        initial_w = self.initial_affine(initial_w)

        mu, logstd = initial_w.split(2, dim=-1)
        w_t = self.generate_samples(mu, logstd)
        z_t = self.initial_noise_to_latent(w_t)

        latents = [z_t]
        dists = [initial_w]
        for t in range(1, seq_len):
            noise_dist = self.inferece(
                torch.cat([transition_parameters[t], z_t, u[t]], dim=-1)
            )
            dists.append(noise_dist)
            w_t = self.generate_samples(*noise_dist.split(2, dim=-1))
            z_t = self.transition_model(z_t, actions[t], w_t)
            latents.append(z_t)

        latents = torch.stack(latents, dim=0)
        dists = torch.stack(dists, dim=0)

        observation_preds = self.decoder(latents.view(seq_len * batch_size, -1)).view(
            *observations.shape
        )
        rec_loss = nn.functional.mse_loss(observation_preds, observations)

        mu, logstd = dists.split(2, dim=-1)
        kl_loss = -logstd + self.anneal_rate * (
            torch.exp(2 * logstd).clamp(1e-5) + mu ** 2
        )
        kl_loss = kl_loss.mean()

        loss = rec_loss + self.kl_weight * kl_loss

        if logger is not None:
            logger["loss/loss"].append(loss.item())
            logger["loss/kl_loss"].append(kl_loss.item())
            logger["loss/rec_loss"].append(rec_loss.item())

    def generate_samples(self, mu, logstd):
        std = torch.exp(logstd).clapm(1e-6)
        samples = torch.randn_like(mu)

        return samples * std + mu


class TransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, noise_dim, **kwargs):
        """Abstract class for transition models.
        All transition models should be called <name>TransitionModel.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim

    def forward(self, latent, action, noise):
        """
        Implements transition in latent space.

        Args:
             latent : z_t
             action : u_t
             noise : w_t

        Returns:
            Next latent z_{t+1}.
        """
        pass


class LocallyLinearTransitionModel(TransitionModel):
    def __init__(
        self, num_matrices, latent_dim, action_dim, noise_dim, hidden_size=16, **kwargs
    ):
        super().__init__(
            latent_dim=latent_dim, action_dim=action_dim, noise_dim=noise_dim
        )

        self.num_matrices = num_matrices

        self.latent_matrix = nn.Parameter(
            torch.randn(num_matrices, latent_dim, latent_dim) * (1.0 / latent_dim)
        )
        self.action_matrix = nn.Parameter(
            torch.randn(num_matrices, action_dim, latent_dim)
            * (1.0 / math.sqrt(latent_dim * action_dim))
        )
        self.noise_matrix = nn.Parameter(
            torch.randn(num_matrices, noise_dim, latent_dim)
            * (1.0 / math.sqrt(latent_dim * noise_dim))
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_matrices),
            nn.Softmax(),
        )

    @staticmethod
    def _get_output(matrix, weight, feature):
        # matrix [num_matrices, dim1, dim2]
        # weight [batch, num_matrices]
        # feature [batch, dim1]
        num_matrices, dim1, dim2 = matrix.size()
        feature_matrix = torch.mm(weight, matrix.view(num_matrices, -1)).view(
            -1, dim1, dim2
        )
        return torch.sum(feature_matrix * feature.unsqueeze(-1), dim=1)

    def forward(self, latent, action, noise):
        weights = self.net(torch.cat([latent, action], dim=-1))

        latent_out = self._get_output(self.latent_matrix, weights, latent)
        action_out = self._get_output(self.action_matrix, weights, action)
        noise_out = self._get_output(self.noise_matrix, weights, noise)

        return latent_out + action_out + noise_out
