import argparse
import gym
import numpy as np
import os
from itertools import count

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical
from torch.autograd import Variable
import wandb
import time
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

TOTAL_EPISODE_STEP = 10000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"



def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)



class Policy(nn.Module):
    def __init__(self, min_act):
        super().__init__()
        self.min_act = min_act
        
        self.main = nn. Sequential(
                        nn.Linear(8, 128),
                        nn.Dropout(p=0.6),
                        nn.Linear(128, 512)
                    )

        self.v = nn.Linear(512, 1, 1)
        self.policy = nn.Linear(512, min_act)
        self.Softmax = nn.Softmax(dim = -1)

        for p in self.main.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

        init.kaiming_uniform_(self.v.weight, a=1.0)
        init.kaiming_uniform_(self.policy.weight, a=1.0)
        self.v.bias.data.zero_()
        self.policy.bias.data.zero_()

    def forward(self, x):
        y = self.main(x)
        value = self.v(y).squeeze()
        pi = self.Softmax(self.policy(y).view(-1, self.min_act))
        return value, pi


class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.

    It is true that LunarLander-v2 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('LunarLander-v2')
        state = self.env.reset()

    def run_episode(self, agent_rref, n_steps):
        """
        Run one episode of n_steps.

        Args:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        state,*_ = self.env.reset()
        for step in range(n_steps):
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)
            # apply the action to the environment, and get the reward
            state,reward, terminated, truncated,_ = self.env.step(action)

            # report the reward & state to the agent for training purpose
            _remote_method(Agent.report, agent_rref, self.id, (reward, state, 1 - (terminated * 1)))

            if terminated or truncated:
                break

class Agent:
    def __init__(self, world_size, args):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.saved_log_probs = {}
        self.saved_cache = {}
        self.saved_entropies = {}
        self.states = {}
        self.rewards = {}
        self.mask = {}

        self.clip_grad_norm = args.clip_grad_norm
        n_actions = gym.make('LunarLander-v2').action_space.n
        self.init_lr = args.lr
        self.total_episode_rewards = np.zeros((world_size))
        self.emulator_steps = np.zeros((world_size))
        self.global_step = 0
        self.running_reward = 0
        self.max_train_steps = args.max_train_steps
        
        self.policy = Policy(n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0


        wandb.login()
        # Log some basic details about the environment
        self.run = wandb.init(project="HPPC Project", ## Name of the project 
                config = {
                "num_workers": args.num_workers,
                "game_name":args.env_name,
                "max_train_steps":args.max_train_steps,
                },
                name=f"Distributed Training with {args.num_workers} environments" # Name of the session
                )

        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.states[ob_info.id] = []
            self.mask[ob_info.id] = []

            self.saved_log_probs[ob_info.id] = []
            self.saved_entropies[ob_info.id], self.saved_cache[ob_info.id] = [], []

    def select_action(self, ob_id, state):
        """
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """
        value, pi = self.policy(torch.from_numpy(state).float())
        self.saved_cache[ob_id].append(value)
        action = pi.data.multinomial(1)

        log_pi  =  (pi + 1e-12).log()
        log_prob = log_pi.gather(1, action).squeeze()
        entropy = -(log_pi * pi).sum(1)

        self.saved_log_probs[ob_id].append(log_prob)
        self.saved_entropies[ob_id].append(entropy)

        return action.item()

    def report(self, ob_id, elements):
        r"""
        Observers call this function to report rewards.
        """
        reward, state, mask = elements
        self.rewards[ob_id].append(reward)
        self.states[ob_id].append(state)
        self.mask[ob_id].append(mask)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each oberser to run n_steps.
        """
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        """
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """ 
        # joins probs and rewards from different observers into lists
        probs, rewards, states, cache, \
            logprobs, entropies, mask = [], [], [], [], [], [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])
            states.append(self.states[ob_id])
            cache.extend(self.saved_cache[ob_id])
            logprobs.extend(self.saved_log_probs[ob_id])
            entropies.extend(self.saved_entropies[ob_id])
            mask.extend(self.mask[ob_id])

        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward
    
        for ob_id in self.rewards:
            self.total_episode_rewards[ob_id] += sum(self.rewards[ob_id])    
            self.emulator_steps[ob_id] += len(self.rewards[ob_id])
            self.global_step += len(self.rewards[ob_id]) 

            if not all(self.mask[ob_id]):
                wandb.log({"Reward":self.total_episode_rewards[ob_id], "Training Steps": self.emulator_steps[ob_id]},self.global_step)
                self.total_episode_rewards[ob_id] = 0
                self.emulator_steps[ob_id] = 0

            self.rewards[ob_id] = []
            self.states[ob_id]  = []
            self.mask[ob_id]    = []
            self.saved_log_probs[ob_id] = []
            self.saved_entropies[ob_id] = []
            self.saved_cache[ob_id] = []
            
        states, mask, rewards =  np.vstack(states), np.array(mask), np.array(rewards).clip(-1,1)
        value, _ = self.policy(torch.from_numpy(states).float())

        R = Variable(value.data.clone())
        rewards, mask = Variable(torch.from_numpy(rewards)), Variable(torch.from_numpy(mask))
        critic_loss  = 0.0
        actor_loss   = 0.0
        entropy_loss = 0.0

        for i in reversed(range(len(rewards))):
            R = rewards[i] + 0.99 * mask[i] * R
            advantage = R - cache[i]
            critic_loss += advantage.pow(2).mul(0.5).mean()
            actor_loss -= logprobs[i].mul(advantage.detach()).mean()
            entropy_loss -= entropies[i].mean()

        

        total_loss =  actor_loss + entropy_loss.mul(0.02) + (critic_loss * 0.5) 
        total_loss = total_loss.mul(1/(len(rewards))) 
        
        # adjust learning rate
        new_lr = self.init_lr - (self.global_step/self.max_train_steps) * self.init_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)  
        self.optimizer.step()

    def finish_run(self):
        self.run.finish()

def run_worker(rank, world_size, args):
    """
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
        agent = Agent(world_size, args)
        global_step_start = 0
        counter = 0
        start_time = time.time()
        n_steps = int(TOTAL_EPISODE_STEP / (world_size - 1))
        while agent.global_step < args.max_train_steps:
            loop_start_time = time.time()
            agent.run_episode(n_steps=n_steps)
            agent.finish_episode()
            counter += 1
            if counter % (1024/world_size) == 0 :
                curr_time = time.time()
                logging.info(f"Ran {agent.global_step} steps, at {(agent.global_step - global_step_start) / (curr_time - start_time)} steps/s, with running reward {agent.running_reward}")
                
                # torch.save(agent.policy.state_dict(), f'./saved_models_dist/model_{args.env_name}_{agent.global_step}.pth')

        agent.finish_run()
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from agents
    rpc.shutdown()


def main():
    
    parser = argparse.ArgumentParser(description='parameters_setting')
    parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers(default: 4)')
    parser.add_argument('--env-name', default='Breakout-v4', metavar='ENV',
                        help='environment to train on (default: BreakoutDeterministic-v4)')
    parser.add_argument('--max-train-steps', type=int, default=80000000, metavar='MS',
                        help='max training step to train PAAC (default: 500000)')
    parser.add_argument('--clip-grad-norm', type=int, default=3.0, metavar='MS',
                        help='globally clip gradient norm(default: 3.0)')

    
    
    arg = parser.parse_args()
    mp.spawn(
        run_worker,
        args=(arg.num_workers, arg),
        nprocs=arg.num_workers,
        join=True
    )

if __name__ == '__main__':
    main()