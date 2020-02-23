import numpy

def handleTime(elapsedTime):
	# better format the elapsedTime into hour,min,sec,msec
	timeString = "{} hour {} min {} sec {} msec".format(elapsedTime // 3600, (elapsedTime % 3600) // 60,
														int(elapsedTime % 60), int(1000 * (elapsedTime % 1)))
	return timeString

def writeFile(result: dict):
	'''
	result = {
		"initMap": start_map_copy,
		"expandNodeCount": nodes_expanded_total,
		"elapsedTime": elapsed_time,
		"branchingFactor": np.mean(branching_factors),
		"score": best_solution.score(),
		"timeToBest": best_solution.final_score_time()
		"finalMap"
	}
	'''
	outputPath = "Output.txt"
	with open(outputPath, "w") as f:
		f.write("Final Score: " + str(result["score"]))
		f.write("\n")
		f.write("Time to reach score: " + handleTime(result["timeToBest"]))
		f.write("\n")
		f.write("Final Map:\n")
		f.write(str(result["finalMap"]))
	
	print("OK everything is done now! Please check " + outputPath + " for more detail!")
	return True