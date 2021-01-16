import signal
import os, time
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from carla import Client
from env import CarlaEnv
from stable_baselines.common.env_checker import check_env
from waypoint_agent import WaypointAgent
from leaderboard.utils.route_indexer import RouteIndexer

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

def get_route_indexer(args, agent):
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    for ri in range(len(route_indexer._configs_list)):
        route_indexer.get(ri).agent = agent
    return route_indexer

def get_route_config(route_indexer, idx=None, empty=False):
    num_configs = len(route_indexer._configs_list)
    assert num_configs > 0, 'no configs in route indexer'
    if idx is None:
        config = route_indexer.get(np.random.randint(num_configs))
    else:
        assert 0 <= idx < num_configs, 'route config index out of range'
        config = route_indexer.get(idx)
    rconfig = {'config': config, 'extra_args': {'empty': empty}}
    return rconfig


def train(args, env, agent):

    # move inside of env?
    route_indexer = get_route_indexer(args, agent) 

    episode_rewards = []
    episode_policy_losses = []
    episode_value_losses = []
    episode_entropies = []

    save_dict = {
            'rewards' : episode_rewards, 
            'policy_losses' : episode_policy_losses,
            'value_losses' : episode_value_losses,
            'entropies' : episode_entropies}

    total_reward = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    episode_steps = 0

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_path = f'leaderboard/results/rl/waypoint_agent/{date_str}'
    os.makedirs(base_save_path)
    os.makedirs(f'{base_save_path}/weights')

    # start environment and run
    obs = env.reset(get_route_config(route_indexer, empty=args.empty))
    for step in tqdm(range(args.total_timesteps)):

        # random exploration at the beginning
        burn_in = step < args.burn_timesteps
        action = agent.predict(obs, burn_in=burn_in)
        new_obs, reward, done, info = env.step(action)
        total_reward += reward
        episode_steps += 1
        #print(reward, done, info)

        # store in replay buffer
        agent.model.replay_buffer.add(obs, action, reward, new_obs, float(done))

        if done or episode_steps > 6000:
            episode_rewards.append(total_reward)
            episode_policy_losses.append(total_policy_loss/episode_steps)
            episode_value_losses.append(total_value_loss/episode_steps)
            episode_entropies.append(total_entropy/episode_steps)

            total_reward = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            episode_steps = 0

            # cleanup and reset
            env.cleanup()
            rconfig = get_route_config(route_indexer, empty=args.empty)
            obs = env.reset(rconfig)
        
        # train at this timestep if applicable
        if step % args.train_frequency == 0 and not burn_in:
            mb_info_vals = []
            for grad_step in range(args.gradient_steps):

                # policy and value network update
                frac = 1.0 - step/args.total_timesteps
                lr = agent.model.learning_rate*frac
                train_vals = agent.model._train_step(step, None, lr)
                policy_loss, _, _, value_loss, entropy, _, _ = train_vals

                # target network update
                if step % args.target_update_interval == 0:
                    agent.model.sess.run(agent.model.target_update_op)

                if step % args.log_frequency == 0:
                    write_str = f'\nstep {step}\npolicy_loss = {policy_loss:.3f}\nvalue_loss = {value_loss:.3f}\nentropy = {entropy:.3f}'
                    tqdm.write(write_str)

        # save model if applicable
        if step % args.save_frequency == 0 and not burn_in:
            weights_path = f'{base_save_path}/weights/{step:07d}'
            agent.model.save(weights_path)

            for name, arr in save_dict.items():
                save_path = f'{base_save_path}/{name}.npy'
                with open(save_path, 'wb') as f:
                    np.save(f, arr)

        obs = new_obs
        

    #print('done training')

def main(args):
    client = Client('localhost', 2000)
    agent_config = {}
    agent_config['mode'] = 'train'
    agent = WaypointAgent(agent_config)
        
    env_config = {}
    env_config['world_port'] = 2000
    env_config['tm_port'] = 8000
    env_config['tm_seed'] = 0

    try:
        env = CarlaEnv(client, agent, env_config)
        #check_env(env)
        train(args, env, agent)
    except KeyboardInterrupt:
        print('caught Ctrl-C')
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        env.cleanup()
        del env
        #client.reload_world()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str)
    parser.add_argument('--scenarios', type=str)
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--burn_timesteps' , type=int, default=5000)
    parser.add_argument('--train_frequency', type=int, default=1)
    parser.add_argument('--gradient_steps', type=int, default=1)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--save_frequency', type=int, default=500)
    parser.add_argument('--log_frequency', type=int, default=500)
    parser.add_argument('--empty', type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
