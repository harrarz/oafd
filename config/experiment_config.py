"""
Algorithm configuration profiles for different optimization strategies.
"""
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2

# Varied configuration profiles for NSGA-II algorithm
NSGA2_PROFILES = {
    'baseline': {
        'population_size': 80,
        'offspring_size': 40,
        'mutation_intensity': 1.0,
        'crossover_rate': 0.9
    },
    'exploratory': {
        'population_size': 100,
        'offspring_size': 80,
        'mutation_intensity': 2.0,
        'crossover_rate': 0.8
    },
    'exploitative': {
        'population_size': 80,
        'offspring_size': 20,
        'mutation_intensity': 0.5,
        'crossover_rate': 0.95
    },
    'mutation_heavy': {
        'population_size': 80,
        'offspring_size': 40,
        'mutation_intensity': 3.0,
        'crossover_rate': 0.9
    }
}

# Varied configuration profiles for SPEA2 algorithm
SPEA2_PROFILES = {
    'baseline': {
        'population_size': 80,
        'offspring_size': 40,
        'mutation_intensity': 1.0,
        'crossover_rate': 0.9
    },
    'expanded_pool': {
        'population_size': 120,
        'offspring_size': 60,
        'mutation_intensity': 1.0,
        'crossover_rate': 0.9
    },
    'exploratory': {
        'population_size': 100,
        'offspring_size': 80,
        'mutation_intensity': 2.0,
        'crossover_rate': 0.8
    },
    'exploitative': {
        'population_size': 80,
        'offspring_size': 40,
        'mutation_intensity': 0.5,
        'crossover_rate': 0.95
    }
}

# Algorithm registry mapping strategy names to algorithm classes and profiles
OPTIMIZER_REGISTRY = {
    'nsga2': (NSGAII, NSGA2_PROFILES),
    'spea2': (SPEA2, SPEA2_PROFILES)
}