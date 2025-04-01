"""
Main entry point for multi-objective optimization experiments.
"""
import os
import multiprocessing
import logging
import numpy as np

from sklearn.datasets import load_iris

from core.utils import secure_print
from output.manager import OutputManager
from experiments.runner import execute_parallel_experiments
from experiments.baseline import run_baseline_reference
from config.experiment_config import DEFAULT_MAX_ITERATIONS, DEFAULT_NUM_RUNS, DEFAULT_GROUP_COUNT, OUTPUT_DIR

# Reduce verbosity of external framework logs
logging.getLogger('jmetal').setLevel(logging.WARNING)


def main():
    """
    Main entry point for the application.
    Loads data, runs experiments and saves results.
    """
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load sample data for demonstration
    secure_print("Loading data...")
    data = load_iris()
    matrix = data.data
    secure_print(f"Data loaded: {matrix.shape[0]} samples with {matrix.shape[1]} features")

    # Run baseline reference for comparison
    secure_print("Running baseline reference...")
    baseline_results = run_baseline_reference(matrix, group_count=DEFAULT_GROUP_COUNT)

    # Execute parallel experiments
    secure_print("Starting parallel experiments...")
    output_manager = OutputManager()
    max_workers = min(multiprocessing.cpu_count(), 4)  # Limit to avoid overloading
    
    execute_parallel_experiments(
        matrix, 
        output_manager,
        n_runs=DEFAULT_NUM_RUNS, 
        max_workers=max_workers,
        group_count=DEFAULT_GROUP_COUNT,
        max_iterations=DEFAULT_MAX_ITERATIONS
    )

    # Create summary directory
    os.makedirs(os.path.join(OUTPUT_DIR, "summary"), exist_ok=True)
    
    # Save a basic summary CSV (can be extended as needed)
    with open(os.path.join(OUTPUT_DIR, "summary", "experiment_summary.txt"), "w") as f:
        f.write(f"Experiments completed with {DEFAULT_NUM_RUNS} runs per configuration\n")
        f.write(f"Total configurations: {len(output_manager.collected_results)}\n")
        f.write(f"Baseline silhouette score: {baseline_results['mean_silhouette']:.4f} ± {baseline_results['std_silhouette']:.4f}\n")

    secure_print("All tasks completed!")


if __name__ == "__main__":
    main()