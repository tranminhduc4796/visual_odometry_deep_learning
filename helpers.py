# Script containing helper functions used in a few other files
import math
import subprocess


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
