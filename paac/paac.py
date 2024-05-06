import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
from torch.multiprocessing import Process, Pipe
from worker import Worker
from model import NatureNetwork as paac_ff
import gym
import numpy as np
## Changes are made here  ----------------------------> SP
import logging
import time
import sys
import wandb

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
def train(args):
	# Wandb initialize
	wandb.login()

	## Log some basic details about the environment
	run = wandb.init(project="HPPC Project", ## Name of the project 
			config = {
			"num_envs": args.num_envs,
			"num_workers": args.num_workers,
			"game_name":args.env_name,
			"max_train_steps":args.max_train_steps,
			},
			name=f"Training with {args.num_envs * args.num_workers} environments" # Name of the session
			)
	logging.debug("Starting training.....")
	torch.multiprocessing.set_start_method('forkserver')
	num_envs = args.num_envs
	num_workers = args.num_workers
	total_envs = num_workers * num_envs
	game_name = args.env_name
	max_train_steps = args.max_train_steps
	n_steps = args.n_steps
	init_lr = args.lr
	clip_grad_norm = args.clip_grad_norm
	num_action = gym.make(game_name).action_space.n
	image_size = 84
	n_stack = 4
	

	model = paac_ff(min_act=num_action).cuda()
	x = Variable(torch.zeros(total_envs, n_stack, image_size, image_size), requires_grad = False).cuda()
	xs = [Variable(torch.zeros(total_envs, n_stack, image_size, image_size)).cuda() for i in range(n_steps)]
	share_reward = [Variable(torch.zeros(total_envs)).cuda() for _ in range(n_steps)]
	share_mask = [Variable(torch.zeros(total_envs)).cuda() for _ in range(n_steps)]

	counter = 0
	optimizer = optim.Adam(model.parameters(), lr=init_lr)

	workers, parent_conns, child_conns = [], [], []
	for i in range(num_workers):
		parent_conn, child_conn = Pipe()
		w = Worker(i, num_envs, game_name, n_stack, child_conn, args)
		w.start()
		workers.append(w)
		parent_conns.append(parent_conn)
		child_conns.append(child_conn)

	new_s = np.zeros((total_envs, n_stack, image_size, image_size))
	total_episode_rewards = [0] * total_envs
	emulator_steps = [0] * total_envs
	start_time = time.time()
	total_rewards = []
	global_step_start = 0
	global_step = global_step_start

	while global_step < max_train_steps:

		loop_start_time = time.time()
		cache_v_series, entropies, sampled_log_probs = [], [], []
		for step in range(n_steps):
			
			## Actor network time 
			act_start = time.time()
			xs[step].data.copy_(torch.from_numpy(new_s))
			v, pi = model(xs[step])
			cache_v_series.append(v)

			sampling_action = pi.data.multinomial(1)

			log_pi = (pi+1e-12).log()
			entropy = -(log_pi*pi).sum(1)
			sampled_log_prob = log_pi.gather(1, Variable(sampling_action)).squeeze()
			sampled_log_probs.append(sampled_log_prob)
			entropies.append(entropy)
			
			send_action = sampling_action.squeeze().cpu().numpy()
			send_action = np.split(send_action, num_workers)
			act_end = time.time() - act_start

			## Calculate Agent-Env Interaction
			envint_start = time.time()
			# send action and then get state
			for parent_conn, action in zip(parent_conns, send_action):
				parent_conn.send(action)
			
			batch_s, batch_r, batch_mask = [], [], []
			for parent_conn in parent_conns:
				s, r, mask = parent_conn.recv()
				batch_s.append(s)
				batch_r.append(r)
				batch_mask.append(mask)

			new_s = np.vstack(batch_s)
			r_disp = np.hstack(batch_r).copy() 
			r = np.hstack(batch_r).clip(-1, 1) # clip reward
			mask = np.hstack(batch_mask)

			for envstep, (done, reward) in enumerate(zip(mask, r_disp)):
				total_episode_rewards[envstep] += reward
				
				emulator_steps[envstep] += 1
				global_step += 1

				if not done: 
					total_rewards.append(total_episode_rewards[envstep])
					wandb.log({"Reward":total_episode_rewards[envstep], "Training Steps": emulator_steps[envstep]},global_step)
		
					total_episode_rewards[envstep] = 0
					emulator_steps[envstep] = 0

			share_reward[step].data.copy_(torch.from_numpy(r))
			share_mask[step].data.copy_(torch.from_numpy(mask))
			envint_end = time.time() - envint_start


		## Critic Network time
		crit_start = time.time()
		x.data.copy_(torch.from_numpy(new_s))
		v, _ = model(x) # v is volatile
		R = Variable(v.data.clone())
		v_loss = 0.0
		policy_loss = 0.0
		entropy_loss = 0.0
		
		for i in reversed(range(n_steps)):

			R =  share_reward[i] + 0.99 * share_mask[i] * R
			advantage = R - cache_v_series[i]
			v_loss += advantage.pow(2).mul(0.5).mean()

			policy_loss -= sampled_log_probs[i].mul(advantage.detach()).mean()
			entropy_loss -= entropies[i].mean()
		
		total_loss = policy_loss + entropy_loss.mul(0.02) +  v_loss*0.5
		total_loss = total_loss.mul(1/(n_steps))

		# adjust learning rate
		new_lr = init_lr - (global_step/max_train_steps)*init_lr
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		
		optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

		optimizer.step()
		crit_end = time.time() - crit_start

		counter += 1
		if counter % (2048 / total_envs) == 0 :
			curr_time = time.time()
			last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
			
			percentEnvInt = envint_end / (curr_time - loop_start_time)
			percentModInt = (crit_end + act_end) / (curr_time - loop_start_time)
			wandb.log({"Environment Interaction":percentEnvInt, "Model Interaction": percentModInt},global_step)
			logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
							.format(global_step,
									n_steps * total_envs / (curr_time - loop_start_time),
									(global_step - global_step_start) / (curr_time - start_time),
									last_ten))
			torch.save(model.state_dict(), f'./saved_models/model_{game_name}_{global_step}.pth')


	run.finish()
	for parent_conn in parent_conns:
		parent_conn.send(None)

	for w in workers:
		w.join()

