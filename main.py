#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  28 09:36:05 2019

@author: farismismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";   # My NVIDIA GTX 1080 Ti FE GPU

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tick

from environment import radio_environment
from QLearningAgent import QLearningAgent as QLearner  # Tabular
#from DQNLearningAgent import DQNLearningAgent as QLearner # Deep with GPU

MAX_EPISODES = 40000 # Succ: [9183, 31481, 32276, 32978, 34448]
MAX_EPISODES_DEEP = 30000 # Succ: []

os.chdir('/Users/farismismar/Desktop/DRL')

seed = 0 

random.seed(seed)
np.random.seed(seed)
 
env = radio_environment(seed=seed)
agent = QLearner(seed=seed)

def run_agent_deep(env, plotting=False):
    max_episodes_to_run = MAX_EPISODES_DEEP # needed to ensure epsilon decays to min
    max_timesteps_per_episode = 20 # one AMR frame ms.
    successful = False
    episode_successful = [] # a list to save the good episodes
    
    losses = []
    future_rewards = []
    replayed_episodes = []
    
    batch_size = 32

    print('Ep.         | TS | Recv. SINR (sv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr')
    print('--'*44)
    
    # Implement the Q-learning algorithm
    for episode_index in np.arange(max_episodes_to_run + 1):
        observation = env.reset()
        observation = np.reshape(observation, [1, agent._state_size])
        agent.begin_episode(observation)
        reward = 0
        done = False
        actions = []
        
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []
        
        for timestep_index in range(max_timesteps_per_episode):
            print('{}/{} | {} | '.format(episode_index, max_episodes_to_run, timestep_index), end='')
            # Perform the power control action and observe the new state.
            action = agent.act(observation)
            
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            next_observation = np.reshape(next_observation, [1, agent._state_size])
            
            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)
            
            # make next_state the new current state for the next frame.
            observation = next_observation

            # If the episode has ended prematurely, penalize the agent.
            if done and timestep_index < max_timesteps_per_episode - 1:
                reward = env.reward_min
            
            if reward < 0:
                successful = False
                
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()

            successful = (reward >= env.reward_max)
                
            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            sinr_ue2_progress.append(env.received_ue2_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            interfering_tx_power_progress.append(env.interfering_transmit_power_dBm)
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            replayed_episodes.append(episode_index)
            
          #  print('At t = {}, action played was {} with reward {}.'.format(timestep_index, action, reward))
                    
        if (successful):
            episode_successful.append(episode_index)

#        print('SINR progress')
#        print(sinr_progress)
#        print('Serving BS transmit power progress')
#        print(serving_tx_power_progress)
#        print('Interfering BS transmit power progress')
#        print(interfering_tx_power_progress)
        
        # Get the performance of the episode z
        q_z, l_z = agent.get_performance()
        losses.append(np.mean(l_z)) # average across the batch size
        future_rewards.append(np.mean(q_z))

#        print(losses) # is equal to the number of episodes + 1
#        print(future_rewards)
 
        # Plot the episode...
        if (plotting) :# and episode_index in [4498]):
            plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, max_episodes_to_run)
            plot_actions(actions, max_timesteps_per_episode, episode_index, max_episodes_to_run)
    
    plot_performance_function_deep(losses, is_loss=True)
    plot_performance_function_deep(future_rewards, is_loss=False)
              
    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    print('Successful episode(s)')
    print(episode_successful)
    agent.save('deep_rl.model')
    
##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##

def run_agent(env, plotting=False):
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = 20 # one AMR frame ms.
    successful = False
    episode_successful = [] # a list to save the good episodes
    Q_values = []

    print('Ep.         | TS | Recv. SINR (sv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr')
    print('--'*44)
    
    # Implement the Q-learning algorithm
    for episode_index in np.arange(max_episodes_to_run + 1):
        observation = env.reset()
        action = agent.begin_episode(observation)
        reward = 0
        done = False
        actions = [action]
        
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []
        
        for timestep_index in range(max_timesteps_per_episode):
            print('{}/{} | {} | '.format(episode_index, max_episodes_to_run, timestep_index), end='')
            # Perform the power control action and observe the new state.
            observation, reward, done, abort = env.step(action)

            # If the episode has ended prematurely, penalize the agent.
            if done and timestep_index < max_timesteps_per_episode - 1:
                reward = env.reward_min
            
            if reward < 0:
                successful = False
                
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()

            successful = (reward >= env.reward_max)
                
            action = agent.act(observation, reward)

            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            sinr_ue2_progress.append(env.received_ue2_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            interfering_tx_power_progress.append(env.interfering_transmit_power_dBm)
            
          #  print('At t = {}, action played was {} with reward {}.'.format(timestep_index, action, reward))
            
        Q_values.append(agent.get_performance())
        
        if (successful):
            episode_successful.append(episode_index)

#        print('SINR progress')
#        print(sinr_progress)
#        print('Serving BS transmit power progress')
#        print(serving_tx_power_progress)
#        print('Interfering BS transmit power progress')
#        print(interfering_tx_power_progress)
        
        # Plot the episode...
        if (plotting and episode_index in [9183, 31481, 32276, 32978, 34448]):
            plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, max_episodes_to_run)
            plot_actions(actions, max_timesteps_per_episode, episode_index, max_episodes_to_run)
  
    plot_performance_function(Q_values)
              
    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    print('Successful episode(s)')
    print(episode_successful)


def plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, max_episodes_to_run):
    # Do some nice plotting here
    fig = plt.figure(figsize=(7,5))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('Transmit Time Interval (1 ms)')
    
    # Only integers                                
    ax = fig.gca()
    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))
    
    ax_sec = ax.twinx()
    
    ax.set_autoscaley_on(False)
    ax_sec.set_autoscaley_on(False)
    
    ax.plot(sinr_progress, marker='o', color='b', label='SINR for UE 0')
    ax.plot(sinr_ue2_progress, marker='*', color='m', label='SINR for UE 2')
    ax_sec.plot(serving_tx_power_progress, linestyle='--', color='k', label='Serving BS')
    ax_sec.plot(interfering_tx_power_progress, linestyle='--', color='c', label='Interfering BS')
    
    ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode)
    
    ax.axhline(y=env.min_sinr, xmin=0, color="red", linewidth=1.5, label='SINR min')
    ax.axhline(y=env.sinr_target, xmin=0, color="green",  linewidth=1.5, label='SINR target')
    ax.set_ylabel('DL Received SINR (dB)')
    ax_sec.set_ylabel('BS Transmit Power (dBm)')
    
    ax.set_ylim(-7,30)
    ax_sec.set_ylim(0,50)
    
    ax.legend(loc="lower left")
    ax_sec.legend(loc='upper right')
    
    plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index, max_episodes_to_run, agent.exploration_rate))
    plt.grid(True)
    
    plt.savefig('figures/measurements_episode_{}.pdf'.format(episode_index), format="pdf")
    plt.show(block=True)
    plt.close(fig)
    
def plot_actions(actions, max_timesteps_per_episode, episode_index, max_episodes_to_run):
    fig = plt.figure(figsize=(7,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('Transmit Time Interval (1 ms)')
    plt.ylabel('Action Transition')
    
     # Only integers                                
    ax = fig.gca()
    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))
    
    ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode)
    
    ax.set_autoscaley_on(False)
    ax.set_ylim(-1,env.num_actions)
    
    plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index, max_episodes_to_run, agent.exploration_rate))
    plt.grid(True)
        
    plt.step(np.arange(len(actions)), actions, color='b', label='Actions')
    plt.savefig('figures/actions_episode_{}.pdf'.format(episode_index), format="pdf")
    plt.show(block=True)
    plt.close(fig)
    return

def plot_performance_function_deep(values, is_loss=False):
#    print(values)
    title = r'\bf Average $Q$' if not is_loss else r'\bf Episode Loss' 
    y_label = 'Expected Action-Value $Q$' if not is_loss else r'\bf Expected Loss' 
    filename = 'q_function.pdf' if not is_loss else 'losses.pdf'
    fig = plt.figure(figsize=(7,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
   # plt.step(episodes, values, color='k')
    plt.plot(np.arange(MAX_EPISODES_DEEP+1), values, linestyle='-', color='k')
    # plt.plot(values, linestyle='-', color='k')
    plt.grid(True)
    plt.savefig('figures/{}'.format(filename), format="pdf")
    plt.show(block=True)
    plt.close(fig)


def plot_performance_function(values):
#    print(values)
    title = r'\bf Average $Q$'
    y_label = 'Expected Action-Value $Q$'
    filename = 'q_function.pdf'
    fig = plt.figure(figsize=(7,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.plot(np.arange(MAX_EPISODES+1), values, color='k')
    plt.grid(True)
    plt.savefig('figures/{}'.format(filename), format="pdf")
    plt.show(block=True)
    plt.close(fig)
    
########################################################################################
    
#run_agent_deep(env, False)
run_agent(env, True)
#run_agent_fpa(env, False)

########################################################################################
