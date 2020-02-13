from queue import PriorityQueue
from Board import *
import time

def A_Star(start: board):
	# sth to implement here
	frontier = PriorityQueue()
	frontier.put((0, start))
	came_from = dict()
	cost_so_far = dict()
	came_from[start] = None
	cost_so_far[start] = 0
	cur_state = None

	start_time = time.time

	while not frontier.empty():
		cur_state = frontier.get()[1]

		if cur_state.finished():
			break

		for next_state in cur_state.neighbors():
			new_cost = cost_so_far[cur_state] + next_state[1]
			if next_state[0] not in cost_so_far or new_cost < cost_so_far[next_state[0]]:
				cost_so_far[next_state] = new_cost
				priority = new_cost + next_state[0].heuristic()
				frontier.put((priority, next_state))
				came_from[next_state[0]] = cur_state

	end_time = time.time

	path = []
	while cur_state != start:
		path.append(cur_state)
		cur_state = came_from[cur_state]
	path.append(start)
	# please add the time elapsedtime to return statement
	return path.reverse()

def greedyHillClimb():
	# sth to implement here
	pass