import numpy as np
import torch
import torch.utils.data.Dataset as Dataset


class GymEpisodes(Dataset):
    """Samples sequences from random trajectories in gym.env"""

    def __init__(self, env, seq_length, num_episodes=100, max_len=256):
        super().__init__()

        self.env = env
        self.seq_len = seq_length

        self.obs = None
        self.actions = None

        self.num_episodes = num_episodes
        self.max_len = max_len

        self.reload_data()

    def reload_data(self):
        ep = 0
        while ep < self.num_episodes:
            obs = self.env.reset()
            action = self.env.action_space.sample()
            states = [obs]
            actions = [action]
            for t in range(1, self.max_len):
                obs, _, done, _ = self.env.step(action)
                if done:
                    break
                states.append(obs)
                action = self.env.action_space.sample()
                actions.append(action)

            self.obs.append(torch.stack(states, dim=0))
            self.actions.append(torch.stack(actions, dim=0))
            ep += 1

        self.obs = torch.stack(self.obs, dim=0)
        self.actions = torch.stack(self.actions, dim=0)

    def __len__(self):
        return self.max_episodes * (self.max_len // self.seq_len)

    def __getitem__(self, item):
        t_0 = np.random.randint(self.max_len - self.seq_len)
        item = item % self.num_episodes
        return (
            self.obs[item, t_0 : t_0 + self.seq_len],
            self.actions[item, t_0 : t_0 + self.seq_len],
        )
