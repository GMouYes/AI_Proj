'''
the actual q learning file
for the exact search algorithm
'''

#import numpy as np

from expSearch import *

def search(**args):
	'''
	placeholder function
	# TODO: implement details
	'''
	height, width = args["world"].shape

	# expected return: dict of following
	policy = np.zeros((height,width)) # 1,2,3,4 indicating up, down, left, right
	reward = np.zeros((height,width)) # real values
	
	result = {
		"policy": policy,
		"reward": reward,
		# others to be determined
		# time might be important
	}

	return result