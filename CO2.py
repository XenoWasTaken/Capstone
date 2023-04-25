import numpy as np

#Expecting 100 samples over the span of 2 minutes, for each new set compare mean to init_mean
def CO2(curr_stats, init_stats):
    relative_average = curr_stats["average"] - init_stats["average"]
    if relative_average < 0:
        likelihood = 0
    else:
        likelihood = (curr_stats["average"] - init_stats["average"])
    return likelihood


def CO2_init():
    init_readings = collect_readings()
    while True:
        print("Initialized, relocate and press Enter to begin")
        user_input = input()
        while user_input != "\n":
            user_input = input()
        new_readings = collect_readings()
        print("Difference in Mean from initialized space: ", CO2(new_readings, init_readings))
        print("Maximal reading from this location: ", new_readings["max"])

def collect_readings():
    #Begin readings, collect over 2 minutes (may need to slow down to get a consistent amount)
    readings = list()
    #######################################
    #Replace with code for getting CO2 reading
    while len(readings) != 100:
        reading = 1
        readings.append(reading)
    #######################################
    stats = dict()
    stats["average"] = sum(readings)/len(readings)
    stats["max"] = max(readings)
    return stats
