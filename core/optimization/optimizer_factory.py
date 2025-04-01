"""
Factory for creating configured optimizer instances.
"""
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from config.algorithm_profiles import OPTIMIZER_REGISTRY


def create_optimizer(strategy_label, profile, task, max_iterations):
    """
    Create and configure an optimization strategy based on label and profile.
    
    Args:
        strategy_label: Algorithm identifier ('nsga2' or 'spea2')
        profile: Dictionary with algorithm parameters
        task: Optimization task instance
        max_iterations: Maximum number of iterations
        
    Returns:
        Configured optimizer instance
    """
    # Get the optimizer class from the registry
    optimizer_class, _ = OPTIMIZER_REGISTRY[strategy_label]
    
    if strategy_label == 'nsga2':
        base_mutation_rate = 1.0 / task.number_of_variables()
        mutation_rate = base_mutation_rate * profile['mutation_intensity']

        crossover = SBXCrossover(probability=profile['crossover_rate'], distribution_index=20.0)
        mutation = PolynomialMutation(probability=mutation_rate, distribution_index=20.0)

        return optimizer_class(
            problem=task,
            population_size=profile['population_size'],
            offspring_population_size=profile['offspring_size'],
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_iterations)
        )
    else:  # strategy_label == 'spea2'
        base_mutation_rate = 1.0 / task.number_of_variables()
        mutation_rate = base_mutation_rate * profile['mutation_intensity']

        crossover = SBXCrossover(probability=profile['crossover_rate'], distribution_index=20.0)
        mutation = PolynomialMutation(probability=mutation_rate, distribution_index=20.0)

        # Ensure offspring size doesn't exceed population for this strategy
        safe_offspring_size = min(profile['offspring_size'], profile['population_size'])

        return optimizer_class(
            problem=task,
            population_size=profile['population_size'],
            offspring_population_size=safe_offspring_size,
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_iterations)
        )