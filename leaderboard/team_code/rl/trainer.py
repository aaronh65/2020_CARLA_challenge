import signal
import os, sys, time, yaml
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from carla import Client
from env import CarlaEnv

#from stable_baselines import SAC
#from stable_baselines.sac.policies import MlpPolicy
#from null_env import NullEnv
#from stable_baselines.common.env_checker import check_env

from waypoint_agent import WaypointAgent
from leaderboard.utils.route_indexer import RouteIndexer

BASE_SAVE_PATH = os.environ.get('BASE_SAVE_PATH', 0)

def train(config, agent, env):

    episode_rewards = []
    episode_policy_losses = []
    episode_value_losses = []
    episode_entropies = []

    save_dict = {
        'rewards' : episode_rewards, 
        'policy_losses' : episode_policy_losses,
        'value_losses' : episode_value_losses,
        'entropies' : episode_entropies
        }


    total_reward = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    episode_steps = 0

    # start environment and run
    obs = env.reset()
    agent.reset()
    for step in tqdm(range(config['total_timesteps'])):

        # random exploration at the beginning
        burn_in = step < config['burn_timesteps']
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
            obs = env.reset()
            agent.reset()
            #print(sys.getrefcount(agent))
        
        # train at this timestep if applicable
        if step % config['train_frequency'] == 0 and not burn_in:
            mb_info_vals = []
            for grad_step in range(config['gradient_steps']):

                # policy and value network update
                frac = 1.0 - step/config['total_timesteps']
                lr = agent.model.learning_rate*frac
                train_vals = agent.model._train_step(step, None, lr)
                policy_loss, _, _, value_loss, entropy, _, _ = train_vals

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy

                # target network update
                if step % config['target_update_interval'] == 0:
                    agent.model.sess.run(agent.model.target_update_op)

                if config['verbose'] and step % config['log_frequency'] == 0:
                    write_str = f'\nstep {step}\npolicy_loss = {policy_loss:.3f}\nvalue_loss = {value_loss:.3f}\nentropy = {entropy:.3f}'
                    tqdm.write(write_str)

        # save model if applicable
        if step % config['save_frequency'] == 0 and not burn_in:
            weights_path = f'{BASE_SAVE_PATH}/weights/{step:07d}'
            agent.model.save(weights_path)

            for name, arr in save_dict.items():
                save_path = f'{BASE_SAVE_PATH}/{name}.npy'
                with open(save_path, 'wb') as f:
                    np.save(f, arr)

        obs = new_obs

    #print('done training')

def main(args):
    client = Client('localhost', 2000)

    # get configs
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    env_config = config['env_config']
    sac_config = config['sac_config']

    agent = WaypointAgent(sac_config)

    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    #for ri in range(len(route_indexer._configs_list)):
    #    route_indexer.get(ri).agent = agent

    try:
        env = CarlaEnv(env_config, client, agent, route_indexer)
        train(sac_config, agent, env)
    except KeyboardInterrupt:
        print('caught KeyboardInterrupt')
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        env.cleanup()
        del env
        #client.reload_world()

def parse_args():
    parser = argparse.ArgumentParser()

    # yaml config path if available
    parser.add_argument('--config_path', type=str)

    # Leaderboard args
    parser.add_argument('--routes', type=str)
    parser.add_argument('--scenarios', type=str)
    parser.add_argument('--repetitions', type=int)

    # env args
    parser.add_argument('--empty', action='store_true')

    # SAC args
    parser.add_argument('--load_from', type=str)
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--burn_timesteps' , type=int, default=5000)
    parser.add_argument('--train_frequency', type=int, default=1)
    parser.add_argument('--gradient_steps', type=int, default=1)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--save_frequency', type=int, default=500)
    parser.add_argument('--log_frequency', type=int, default=500)

    # other args
    parser.add_argument('--save_images', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
