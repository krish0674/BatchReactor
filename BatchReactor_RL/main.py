from torch import manual_seed
from source.agent import Agent
from source.critic import Critic
import source.policy as Policy

# Settings and hyperparameters
config = {
    "manual_seed": None,
    "policy_type": "MPC_Policy",  # Choose between "MPC_Policy" and "MLP_Policy"
    "num_environments": 128,
    "short_horizon_length": 8,
    "weight_decay_critic": 1e-5,
    "exploration_noise": 0.1,
}

if config["policy_type"] == "MPC_Policy":
    config["timesteps_total"] = 1_000_000
    config["learning_rate_policy"] = 1e-5
    config["learning_rate_critic"] = 3*1e-5
elif config["policy_type"] == "MLP_Policy":
    config["timesteps_total"] = 5_000_000
    config["learning_rate_policy"] = 1e-4
    config["learning_rate_critic"] = 3*1e-4

def main():
    # Set random seeds for reproducibility
    if config["manual_seed"] is not None:
        manual_seed(config["manual_seed"])

    # Set up environment, policy, critic, and agent
    policy = getattr(Policy, config["policy_type"])()
    critic = Critic()
    agent = Agent(config["num_environments"], config["short_horizon_length"],
                  policy, critic, config["learning_rate_policy"],
                  config["learning_rate_critic"], config["weight_decay_critic"],
                  exploration_noise=config["exploration_noise"])

    # load agent (if you have a pre-trained model you want to test or continue training)
    # agent.load_agent(file_name='best_val_agent')

    # Test untrained policy
    agent.test_policy(
        num_episodes=1, episode_length=int(24*180),
        train_or_test_prices='test',
        plot=True, print_results=True, plot_name_prefix='untrained')

    # Train policy
    agent.train_policy(config["timesteps_total"])

    # Test trained policy
    agent.load_agent(directory='./models/CurrentRun', file_name='best_val_agent')
    agent.test_policy(
        num_episodes=1, episode_length=int(24*365/2),
        train_or_test_prices='test',
        plot=True, print_results=True, plot_name_prefix='trained')

    print('done')

if __name__ == "__main__":
    main()
