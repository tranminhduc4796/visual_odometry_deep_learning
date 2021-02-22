# Script containing helper functions used in a few other files
import math
import subprocess
import os
import torch
import shutil


def first_ge(sorted_list, input_key):
	"""
	Helper method to return the first index in a sorted list that is greater than or equal to the given key
	"""
	# Basically, do a binary search
	low = 0
	high = len(sorted_list)
	while low != high:
		mid = int(math.floor((low + high) / 2))
		if sorted_list[mid] <= input_key:
			low = mid + 1
		else:
			high = mid
	# If an element greater than equal to the key exists
	# both low and mid contain the index of that element
	if low == high:
		return low
	else:
		return -1


"""
Script to query nvidia-smi.
https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
"""


def get_gpu_memory_map():
	"""
	To track GPU usage
	:return: dict: GPU ID and memory usage in MB
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		])
	result = result.decode('utf-8')
	# Convert lines into a dictionary
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map


def init_dir_structure(config, base_dir, exp_dir):
	if not os.path.exists(os.path.join(base_dir, config.cache_dir, config.dataset)):
		os.makedirs(os.path.join(base_dir, config.cache_dir, config.dataset))
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
		print('Created dir: ', exp_dir)
	if not os.path.exists(os.path.join(exp_dir, 'models')):
		os.makedirs(os.path.join(exp_dir, 'models'))
		print('Created dir: ', os.path.join(exp_dir, 'models'))
	if not os.path.exists(os.path.join(exp_dir, 'plots', 'traj')):
		os.makedirs(os.path.join(exp_dir, 'plots', 'traj'))
		print('Created dir: ', os.path.join(exp_dir, 'plots', 'traj'))
	if not os.path.exists(os.path.join(exp_dir, 'plots', 'loss')):
		os.makedirs(os.path.join(exp_dir, 'plots', 'loss'))
		print('Created dir: ', os.path.join(exp_dir, 'plots', 'loss'))
	for seq in range(11):
		if not os.path.exists(os.path.join(exp_dir, 'plots', 'traj', str(seq).zfill(2))):
			os.makedirs(os.path.join(exp_dir, 'plots', 'traj', str(seq).zfill(2)))
			print('Created dir: ', os.path.join(exp_dir, 'plots', 'traj', str(seq).zfill(2)))


def save_checkpoint(state, exp_dir, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join(exp_dir, filename))
	if is_best:
		shutil.copyfile(os.path.join(exp_dir, filename), os.path.join(exp_dir, 'model_best.pth.tar'))


class EarlyStopping:
	def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
		self.mode = mode
		self.min_delta = min_delta
		self.patience = patience
		self.best = None
		self.num_bad_epochs = 0
		self.is_better = None
		self._init_is_better(mode, min_delta, percentage)

		if patience == 0:
			self.is_better = lambda a, b: True
			self.step = lambda a: False

	def step(self, metrics):
		if self.best is None:
			self.best = metrics
			return False

		if torch.isnan(metrics):
			return True

		if self.is_better(metrics, self.best):
			self.num_bad_epochs = 0
			self.best = metrics
		else:
			self.num_bad_epochs += 1

		if self.num_bad_epochs >= self.patience:
			return True

		return False

	def _init_is_better(self, mode, min_delta, percentage):
		if mode not in {'min', 'max'}:
			raise ValueError('mode ' + mode + ' is unknown!')
		if not percentage:
			if mode == 'min':
				self.is_better = lambda a, best: a < best - min_delta
			if mode == 'max':
				self.is_better = lambda a, best: a > best + min_delta
		else:
			if mode == 'min':
				self.is_better = lambda a, best: a < best - (
						best * min_delta / 100)
			if mode == 'max':
				self.is_better = lambda a, best: a > best + (
						best * min_delta / 100)