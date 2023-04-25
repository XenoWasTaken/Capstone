import numpy as np

#Expecting 50 samples over the span of 5 seconds, for each 5 seconds calculate statistics and compare to previous 5
def CO2(readings, past_stats):
    stats = dict()
    hist, bins = np.histogram(readings, bins='auto', density=True)
    stats["highest_spike"] = max([readings[i] - readings[i-1] for i in range(len(readings))])
    stats["average"] = sum(readings)/len(readings)
    stats["entropy"] = -np.sum(hist * np.log2(hist + 1e-6))

    #Compare to previous average
    relative = stats["average"] - past_stats["average"]
    if relative < 0:
        likelihood = 0
    else:
        likelihood = (stats["average"] - past_stats["average"])
    return stats, likelihood


    