import sys
import argparse
from collections import defaultdict

import gym
import tensorboardX
import torch

import dvbf.utils as utils
from dvbf.bayes_filter import BayesFilter, get_transition_model
from dvbf.data_loader import GymEpisodes


def parse_arguments(args_to_parse):
    """Parse command line arguments
    Args:
        args_to_parse : list of str

    Returns:
        argparse.ArgumentParser object
    """
    description = "PyTorch implementation of Deep Variational Bayes Filters."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "name", type=str, help="Name of the model for storing and loading purposes."
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1234,
        help="Random seed. Can be `None` for stochastic behavior.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, help="Checkpoints every N epochs."
    )
    parser.add_argument(
        "--env", type=str, default="Pendulum-v0", help="Environment name"
    )
    parser.add_argument(
        "--seq-length", type=int, default=32, help="Sequence length for training."
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes in memory."
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=256,
        help="Maximum length of an episode.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Minibatch size.")
    parser.add_argument(
        "--latent-dim", type=int, default=4, help="Dimensionality of latent space."
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=4,
        help="Dimensionality of random samples (w_t).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=16,
        help="Dimensionality of hidden layers.",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1.0,
        help="Weight of KL term.",
    )
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm."
    )
    parser.add_argument(
        "--transition-model",
        choices=["locally_linear"],
        default="locally_linear",
        help="Type of latent transition model.",
    )
    parser.add_argument(
        "--num-matrices",
        type=int,
        default=6,
        help="Number of matrices in Locally linear transiton model.",
    )
    args = parser.parse_args(args_to_parse)
    return args


def main(args):
    utils.seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_dir = utils.create_save_dir(args.name)
    logger = utils.get_txt_logger(save_dir)
    tb_writer = tensorboardX.SummaryWriter(save_dir)

    env = gym.make(args.env)
    args.input_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    logger.log("{}\n".format(args))
    utils.save_args(save_dir, args)

    transition_model = get_transition_model(args.transition_model)
    transition_model = transition_model(**vars(args))
    net = BayesFilter(
        transition_model,
        latent_dim=args.latent_dim,
        noise_dim=args.noise_dim,
        action_dim=args.action_dim,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        kl_weight=args.kl_weight,
    )

    model_state = utils.get_model_state(save_dir)
    if model_state is not None:
        net.load_state_dict(model_state)

    net.to(device)

    episodes = GymEpisodes(
        env,
        args.seq_length,
        num_episodes=args.num_episodes,
        max_len=args.max_episode_length,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    for e in range(args.epochs):
        epoch_logger = defaultdict(list)
        for data in episodes:
            observations, actions = data

            # switch from batch-first to time-first
            observations = torch.permute(1, 0, 2).to(device)
            actions = torch.permute(1, 0, 2).to(device)

            loss = net(observations, actions, epoch_logger)

            optimizer.zero_grad()
            loss.backward()
            epoch_logger["grad_norm"].append(utils.grad_norm(net))
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()

        msg = "Epoch {} | ".format(e)
        for k, v in epoch_logger.items():
            msg += k + ": {:.4f}".format(v)
            tb_writer.add_scalar(k, v, e)

        logger.log(msg)
        net.update_annealing()

        if e % args.checkpoint_every == 0:
            utils.save_model_state(save_dir, net)
            logger.log("Checkpoint saved!")
            episodes.reload_data()

    logger.log("Training finished!")


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
