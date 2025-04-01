"""
Experiment execution functions.
"""
import concurrent.futures
from functools import partial

from core.utils import secure_print
from core.metrics import AnalysisMetrics
from core.progress import ProgressTracker
from core.optimization.optimization_task import OptimizationTaskDualCriteria
from core.optimization.optimizer_factory import create_optimizer

from config.algorithm_profiles import OPTIMIZER_REGISTRY
from config.experiment_config import DEFAULT_GROUP_COUNT, DEFAULT_MAX_ITERATIONS


def execute_single_experiment(matrix, strategy_label, profile_label, profile, run_number, 
                             group_count=DEFAULT_GROUP_COUNT, max_iterations=DEFAULT_MAX_ITERATIONS):
    """
    Execute a single experiment with specific strategy/profile combination.
    
    Args:
        matrix: Data matrix of shape (elements_count, attributes_count)
        strategy_label: Algorithm identifier ('nsga2' or 'spea2')
        profile_label: Configuration profile name
        profile: Configuration parameters dictionary
        run_number: Run identification number
        group_count: Number of clusters to form
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary with experiment results
    """
    experiment_id = f"{strategy_label}-{profile_label}-run{run_number}"
    secure_print(f"🚀 Starting {experiment_id}")

    try:
        task = OptimizationTaskDualCriteria(matrix, group_count=group_count)
        optimizer = create_optimizer(strategy_label, profile, task, max_iterations)

        # Progress tracking
        tracker = ProgressTracker()
        optimizer.observable.register(tracker)

        optimizer.run()
        solutions = optimizer.result()

        quality_score = AnalysisMetrics.compute_solution_quality(solutions)

        # Identify best individual solution by quality
        best_candidate = max(solutions, key=lambda s: AnalysisMetrics.compute_solution_quality([s]))

        secure_print(f"✅ Completed {experiment_id}")

        return {
            'id': experiment_id,
            'strategy': strategy_label,
            'profile': profile_label,
            'run': run_number,
            'solutions': solutions,
            'quality': quality_score,
            'best_candidate': best_candidate,
            'best_candidate_quality': AnalysisMetrics.compute_solution_quality([best_candidate]),
            'task': task
        }
    except Exception as e:
        secure_print(f"❌ Error in {experiment_id}: {str(e)}")
        raise


def execute_parallel_experiments(matrix, output_manager, n_runs=10, max_workers=None, 
                                group_count=DEFAULT_GROUP_COUNT, max_iterations=DEFAULT_MAX_ITERATIONS):
    """
    Execute all configured experiments in parallel.
    
    Args:
        matrix: Data matrix of shape (elements_count, attributes_count)
        output_manager: Output manager instance to collect results
        n_runs: Number of repeated runs per configuration
        max_workers: Maximum number of concurrent workers
        group_count: Number of clusters to form
        max_iterations: Maximum number of iterations per experiment
    """
    all_experiments = []
    for strategy_label, (_, profile_dict) in OPTIMIZER_REGISTRY.items():
        for profile_label, profile in profile_dict.items():
            for run_number in range(1, n_runs + 1):
                all_experiments.append((strategy_label, profile_label, profile, run_number))

    total_experiments = len(all_experiments)
    secure_print(f"Preparing to execute {total_experiments} experiments with {max_workers} worker threads")

    completed = 0

    execute_func = partial(
        execute_single_experiment, 
        matrix, 
        group_count=group_count, 
        max_iterations=max_iterations
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(execute_func, strategy, profile_label, profile, run_num)
            for strategy, profile_label, profile, run_num in all_experiments
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                output_manager.add_result(result)
                completed += 1
                if completed % 5 == 0 or completed == total_experiments:
                    secure_print(f"Progress: {completed}/{total_experiments} experiments complete ({completed / total_experiments * 100:.1f}%)")
            except Exception as e:
                secure_print(f"❌ Error in experiment: {e}")

    secure_print("All experiments completed, saving final results...")
    output_manager.save_results(matrix)
    secure_print("✅ All experiments completed!")