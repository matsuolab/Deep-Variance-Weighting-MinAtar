# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
from gym.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import minatar
import numpy as np


class MinAtarEnv(gym.Env):
    def __init__(self, game_name):
        self.env = minatar.Environment(env_name=game_name)
        state_shape = self.env.state_shape()
        state_shape = [state_shape[-1], state_shape[0], state_shape[1]]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=state_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.env.num_actions())

    def _state(self):
        s = self.env.state()
        return np.transpose(s, (2, 0, 1)).astype(np.float32)

    def reset(self):
        self.env.reset()
        return self._state()

    def step(self, action):
        r, terminal = self.env.act(action)
        return self._state(), r, terminal, {}

    def render(self):
        self.env.display_state(10)
        self.env.close_display()


def make_minatar(game_name):
    return TimeLimit(MinAtarEnv(game_name), max_episode_steps=10000)


# settings are same as: https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="minatar",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the experiments. cuda or cpu")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="breakout",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--variance-learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the variance optimizer")
    parser.add_argument("--weight-learning-rate", type=float, default=1e-3,
        help="the learning rate of the weight scaler")
    parser.add_argument("--buffer-size", type=int, default=100000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.1,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--env-random-prob", type=float, default=0.0,
        help="action is randomly executed for adding stochasticity")

    # for Munchausen RL
    parser.add_argument("--kl-coef", type=float, default=0.027,
        help="KL regularization coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.003, 
        help="Entropy regularization coefficient")
    parser.add_argument("--clip-value-min", type=float, default=-1, 
        help="Clipping trick for Munchausen RL")

    # for weighting
    parser.add_argument("--weight-type", choices=["none", "variance-net"], default="none",
        help="weighting type")
    parser.add_argument("--weight-epsilon", type=float, default=0.1,
        help="Small term to avoid numerical issue")
    parser.add_argument("--weight-min", type=float, default=0.1,
        help="Minimum weight value to avoid poor training")

    args = parser.parse_args()
    # fmt: on
    return args

    
class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, random_epsilon):
        super().__init__(env)
        self.random_epsilon = random_epsilon
    
    def action(self, act):
        if random.random() < self.random_epsilon:
            act = self.action_space.sample()
        return act


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = make_minatar(env_id)
        env = RandomActionWrapper(env, args.env_random_prob)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Final fully connected hidden layer:
#   the number of linear unit depends on the output of the conv
#   the output consist 128 rectified units
def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_channels = env.observation_space.shape[1]
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=num_linear_units, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class VarianceNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        in_channels = env.observation_space.shape[1]
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=num_linear_units, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=env.single_action_space.n),
        )

    def forward(self, x):
        variance = torch.exp(self.network(x))
        variance = variance + args.weight_epsilon  # to avoid numerical issue
        return variance


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def compute_dqn_target(net, data, args):
    next_target_max, _ = net(data.next_observations).max(dim=1)
    td_target = data.rewards.flatten() + args.gamma * next_target_max * (1 - data.dones.flatten())
    return td_target


def compute_munchausen_dqn_target(net, data, args):
    kl_coef, ent_coef = args.kl_coef, args.ent_coef
    tau = kl_coef + ent_coef
    alpha = kl_coef / tau

    target = net(data.observations)
    next_target = net(data.next_observations)

    log_pol = nn.functional.log_softmax(target / tau, dim=1)
    log_pol = log_pol.gather(1, data.actions).squeeze()
    munchausen = alpha * torch.clip(tau * log_pol, min=args.clip_value_min)
    next_pol = nn.functional.softmax(next_target / tau, dim=1)
    next_log_pol = nn.functional.log_softmax(next_target / tau, dim=1)
    next_v = (next_pol * (next_target - tau * next_log_pol)).sum(dim=1)
    td_target = munchausen + data.rewards.flatten() + args.gamma * next_v * (1 - data.dones.flatten())
    return td_target, munchausen


def compute_variance_target(target_network, prev_target_network, data, args):
    kl_coef, ent_coef = args.kl_coef, args.ent_coef
    Pv = target_network(data.observations).gather(1, data.actions).squeeze()
    if kl_coef == ent_coef == 0:  # DQN
        Pv_targ = compute_dqn_target(prev_target_network, data, args)
    else:  # Munchausen DQN
        Pv_targ, _ = compute_munchausen_dqn_target(prev_target_network, data, args)
    variance_target = (Pv_targ - Pv) ** 2
    return variance_target


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed, 0, args.capture_video, run_name),  # for evaluation
        make_env(args.env_id, args.seed, 1, args.capture_video, run_name)   # for training
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # for q training
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    for p in target_network.parameters():
        p.requires_grad = False

    # for weight training
    if args.weight_type == "variance-net":
        prev_target_network = QNetwork(envs).to(device)
        prev_target_network.load_state_dict(q_network.state_dict())
        for p in prev_target_network.parameters():
            p.requires_grad = False
        variance_network = VarianceNetwork(envs, args).to(device)
        variance_network_frozen = VarianceNetwork(envs, args).to(device)
        variance_network_frozen.load_state_dict(variance_network.state_dict())
        for p in variance_network_frozen.parameters():
            p.requires_grad = False
        variance_optimizer = optim.Adam(variance_network.parameters(), lr=args.variance_learning_rate)
        log_weight_scaler = torch.ones(1, requires_grad=True, device=device)  # to make the weight average == 1.0
        weight_optimizer = optim.Adam([log_weight_scaler], lr=args.weight_learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        q_values = q_network(torch.Tensor(obs).to(device))

        # action from the policy. saved to the buffer
        actions = torch.argmax(q_values, dim=1).cpu().numpy()  # (greedy action, explore action)
        if random.random() < epsilon:
            actions[1] = envs.single_action_space.sample()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        eval_info = infos[0]
        if "episode" in eval_info.keys():
            print(f"global_step={global_step}, episodic_return={eval_info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", eval_info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", eval_info["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # save only the data from exploration env
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs[1:2], real_next_obs[1:2], actions[1:2], rewards[1:2], dones[1:2], infos[1:2])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        kl_coef, ent_coef = args.kl_coef, args.ent_coef
        tau = kl_coef + ent_coef

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                if args.weight_type == "variance-net":
                    # ===== train variance network ===== 
                    with torch.no_grad():
                        variance_target = compute_variance_target(target_network, prev_target_network, data, args)

                    old_variance = variance_network(data.observations).gather(1, data.actions).squeeze()
                    variance_loss = F.huber_loss(variance_target, old_variance)
                    old_variance_mean = old_variance.mean().item()

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/variance", old_variance_mean, global_step)
                        writer.add_scalar("losses/variance_loss", variance_loss, global_step)
                        writer.add_scalar("losses/variance_target", variance_target.mean().item(), global_step)

                    variance_optimizer.zero_grad()
                    variance_loss.backward()
                    variance_optimizer.step()

                    # ===== train weight_scaler ===== 
                    with torch.no_grad():
                        variance = variance_network_frozen(data.observations).gather(1, data.actions).squeeze()

                    weight_loss = ((log_weight_scaler.exp() / variance).mean() - 1.0) ** 2
                    weight_optimizer.zero_grad()
                    weight_loss.backward()
                    weight_optimizer.step()

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/weight_scaler", log_weight_scaler.exp(), global_step)

                # ===== train q network ===== 
                with torch.no_grad():
                    if kl_coef == ent_coef == 0:  # DQN
                        td_target = compute_dqn_target(target_network, data, args)

                    else:  # Munchausen DQN
                        td_target, munchausen = compute_munchausen_dqn_target(target_network, data, args)
                        if global_step % 100 == 0:
                            writer.add_scalar("munchausen/mean", munchausen.mean().item(), global_step)
                            writer.add_scalar("munchausen/max", munchausen.max().item(), global_step)
                            writer.add_scalar("munchausen/min", munchausen.min().item(), global_step)

                with torch.no_grad():
                    if args.weight_type == "none":
                        weights = torch.ones_like(td_target)
                    else:
                        weights = log_weight_scaler.exp() / variance
                weights = torch.clip(weights, min=args.weight_min)  # to ensure learning from all data
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = torch.mean(weights * (td_target - old_val) ** 2)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("weights/average", weights.mean().item(), global_step)
                    writer.add_scalar("weights/max", weights.max().item(), global_step)
                    writer.add_scalar("weights/min", weights.min().item(), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                if args.weight_type == "variance-net":
                    variance_network_frozen.load_state_dict(variance_network.state_dict())
                    prev_target_network.load_state_dict(target_network.state_dict())
                target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
