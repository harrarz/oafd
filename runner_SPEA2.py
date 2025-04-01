import os
import csv
import logging
import threading
import concurrent.futures
import multiprocessing
from functools import partial
from time import sleep
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import HyperVolume

from OptimizationTaskDualCriteria import OptimizationTaskDualCriteria

# Reduce verbosity of external framework logs
logging.getLogger('jmetal').setLevel(logging.WARNING)

#####################################################
#             Thread Safety Utilities                #
#####################################################

sync_lock = threading.Lock()

def secure_print(*args, **kwargs):
    """
    Thread-safe console output with timestamp.
    Usage: secure_print("Message", key=value, ...)
    """
    with sync_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}]", *args, **kwargs)

#####################################################
#      Algorithm Configuration Registry              #
#####################################################

# Varied configuration profiles for first algorithm type
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

# Varied configuration profiles for second algorithm type
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

OPTIMIZER_REGISTRY = {
    'nsga2': (NSGAII, NSGA2_PROFILES),
    'spea2': (SPEA2, SPEA2_PROFILES)
}

#####################################################
#              Utility Classes                      #
#####################################################

class AnalysisMetrics:
    """
    Utility methods for pattern analysis metrics and operations.
    Provides solution transformation, group assignment, and evaluation metrics.
    """

    @staticmethod
    def transform_solution(task, solution):
        """
        Transform encoded solution to pattern prototype coordinates.

        Args:
            task: The optimization task instance.
            solution: Encoded solution containing flattened prototype coordinates.

        Returns:
            A NumPy array of shape (group_count, attributes_count) for the prototypes.
        """
        group_count = task.group_count
        attributes_count = task.matrix.shape[1]
        return np.array(solution.variables).reshape(group_count, attributes_count)

    @staticmethod
    def assign_groups(matrix, prototypes):
        """
        Assign elements to their nearest prototypes.

        Args:
            matrix: Element data of shape (elements_count, attributes_count).
            prototypes: Prototype coordinates of shape (group_count, attributes_count).

        Returns:
            A NumPy array of group assignments.
        """
        distances = np.zeros((matrix.shape[0], prototypes.shape[0]))
        for i, prototype in enumerate(prototypes):
            distances[:, i] = np.linalg.norm(matrix - prototype, axis=1)
        return np.argmin(distances, axis=1)

    @staticmethod
    def compute_topology_score(matrix, assignments, local_neighbors=10):
        """
        Calculate topology preservation score based on nearest neighbors.

        Args:
            matrix: Element data of shape (elements_count, attributes_count).
            assignments: Group assignments for each element.
            local_neighbors: Number of nearest neighbors to consider.

        Returns:
            The topology preservation score (lower is better).
        """
        elements_count = matrix.shape[0]
        nn_finder = NearestNeighbors(n_neighbors=local_neighbors + 1, metric='euclidean')
        nn_finder.fit(matrix)
        neighbors = nn_finder.kneighbors(matrix, return_distance=False)[:, 1:]

        topology_score = 0.0
        for i in range(elements_count):
            # Decreasing penalties for neighbors with different assignments
            penalties = 1.0 / np.arange(1, local_neighbors + 1)
            different_group_mask = assignments[i] != assignments[neighbors[i]]
            topology_score += np.sum(penalties[different_group_mask])
        return topology_score

    @staticmethod
    def compute_solution_quality(solutions):
        """
        Calculate the hypervolume indicator for a set of solutions.

        Args:
            solutions: A list of solution objects.

        Returns:
            Hypervolume indicator value (float).
        """
        if not solutions:
            return 0.0
        front_criteria = [s.objectives for s in solutions]

        # Determine reference point slightly worse than worst observed values
        criterion1_max = max(obj[0] for obj in front_criteria)
        criterion2_max = max(obj[1] for obj in front_criteria)
        reference_point = [criterion1_max * 1.05, criterion2_max * 1.05]

        hypervolume = HyperVolume(reference_point)
        return hypervolume.compute(front_criteria)


class ProgressTracker:
    """
    Tracks algorithm progress by recording performance metrics over time.
    """

    def __init__(self):
        self.iterations = []
        self.quality_scores = []
        self.criterion1_values = []
        self.criterion2_values = []

    def update(self, **kwargs):
        iteration = kwargs.get("EVALUATIONS", None)
        solutions = kwargs.get("SOLUTIONS", [])
        max_iterations = kwargs.get("MAX_EVALUATIONS", 0)

        if iteration is not None and solutions:
            # Record data at regular intervals or at completion
            if iteration % 5 == 0 or iteration == max_iterations:
                quality = AnalysisMetrics.compute_solution_quality(solutions)
                criterion1_mean = np.mean([s.objectives[0] for s in solutions])
                criterion2_mean = np.mean([s.objectives[1] for s in solutions])

                self.iterations.append(iteration)
                self.quality_scores.append(quality)
                self.criterion1_values.append(criterion1_mean)
                self.criterion2_values.append(criterion2_mean)


class OutputManager:
    """
    Manages experiment output collection, visualization, and persistence.
    """

    def __init__(self):
        self.collected_results = {}
        self.visualizations_queue = []

    def add_result(self, result):
        """
        Register a single experiment result.

        Args:
            result: Dictionary containing experiment outcome data.
        """
        strategy = result['strategy']
        profile = result['profile']

        with sync_lock:
            if (strategy, profile) not in self.collected_results:
                self.collected_results[(strategy, profile)] = {
                    'all_solutions': [],
                    'quality_scores': [],
                    'best_candidates': [],
                    'top_solution': None,
                    'top_quality': -np.inf,
                    'source_task': None
                }

            r = self.collected_results[(strategy, profile)]
            r['all_solutions'].extend(result['solutions'])
            r['quality_scores'].append(result['quality'])
            r['best_candidates'].append((result['best_candidate'], result['best_candidate_quality']))

            if result['best_candidate_quality'] > r['top_quality']:
                r['top_quality'] = result['best_candidate_quality']
                r['top_solution'] = result['best_candidate']
                r['source_task'] = result['task']

    def save_results(self, matrix):
        """
        Save all experiment results to disk with visualizations.

        Args:
            matrix: The original data matrix used in the experiments.
        """
        with sync_lock:
            for (strategy, profile) in self.collected_results.keys():
                os.makedirs(f"output/{strategy}/{profile}", exist_ok=True)

        # Prepare visualizations
        for (strategy, profile), result in self.collected_results.items():
            output_dir = f"output/{strategy}/{profile}"

            top_solution = result['top_solution']
            source_task = result['source_task']

            # Queue visualization tasks
            if top_solution is not None and source_task is not None:
                # Group visualization
                self.visualizations_queue.append(
                    (self._visualize_groups,
                     (source_task, top_solution,
                      os.path.join(output_dir, "top_grouping.png")))
                )
                # Compare with baseline method
                prototypes = AnalysisMetrics.transform_solution(source_task, top_solution)
                self.visualizations_queue.append(
                    (self._compare_with_baseline,
                     (matrix, prototypes, f"{strategy.upper()} - {profile}",
                      os.path.join(output_dir, "comparison_plot.png")))
                )

            # Solution frontier
            self.visualizations_queue.append(
                (self._plot_solution_frontier,
                 (result['all_solutions'], os.path.join(output_dir, "solution_frontier.png")))
            )

            # Quality score distribution
            self.visualizations_queue.append(
                (self._plot_distribution,
                 (result['quality_scores'],
                  os.path.join(output_dir, "quality_distribution.png"),
                  "Quality Score Distribution"))
            )

            # Save CSV data
            self._save_frontier_to_csv(
                result['all_solutions'],
                os.path.join(output_dir, "solution_frontier.csv")
            )
            self._save_best_candidates_to_csv(
                result['best_candidates'],
                os.path.join(output_dir, "best_candidates.csv")
            )

            secure_print(f"✅ Results prepared for {strategy} - {profile}")

        secure_print("Creating visualizations...")
        self.process_visualizations()
        secure_print("All visualizations saved!")

    def process_visualizations(self):
        """Process all queued visualization tasks in batch."""
        for viz_func, args in self.visualizations_queue:
            viz_func(*args)
        self.visualizations_queue = []

    ###################### Visualization Methods ######################

    def _visualize_groups(self, task, solution, save_path):
        """
        Create a 2D projection of the grouping solution using PCA.
        """
        if task is None or solution is None:
            secure_print(f"Warning: Cannot create visualization for {save_path} - missing data")
            return

        prototypes = AnalysisMetrics.transform_solution(task, solution)
        assignments = AnalysisMetrics.assign_groups(task.matrix, prototypes)

        pca = PCA(n_components=2)
        matrix_projected = pca.fit_transform(task.matrix)
        prototypes_projected = pca.transform(prototypes)

        plt.figure(figsize=(6, 5))
        plt.scatter(matrix_projected[:, 0], matrix_projected[:, 1], c=assignments, cmap='viridis', s=30)
        plt.scatter(prototypes_projected[:, 0], prototypes_projected[:, 1], c='red', s=150,
                    marker='x', label='Prototypes')
        plt.title("Top Quality Grouping Solution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _compare_with_baseline(self, matrix, prototypes, method_name, output_path):
        """
        Compare evolutionary solution with baseline (KMeans) on multiple metrics.
        """
        if prototypes is None:
            secure_print(f"Warning: Cannot create comparison for {output_path} - missing prototypes")
            return

        assignments_evo = AnalysisMetrics.assign_groups(matrix, prototypes)

        # Baseline reference method
        kmeans = KMeans(n_clusters=len(prototypes), random_state=42).fit(matrix)
        assignments_baseline = kmeans.labels_

        metrics = {
            'Silhouette': [
                silhouette_score(matrix, assignments_evo),
                silhouette_score(matrix, assignments_baseline)
            ],
            'Compactness': [
                np.sum(np.min(pairwise_distances(matrix, prototypes), axis=1) ** 2),
                np.sum(np.min(pairwise_distances(matrix, kmeans.cluster_centers_), axis=1) ** 2)
            ],
            'Topology': [
                AnalysisMetrics.compute_topology_score(matrix, assignments_evo),
                AnalysisMetrics.compute_topology_score(matrix, assignments_baseline)
            ]
        }

        metric_labels = list(metrics.keys())
        evo_scores = [metrics[k][0] for k in metric_labels]
        baseline_scores = [metrics[k][1] for k in metric_labels]

        x = np.arange(len(metric_labels))
        width = 0.35

        plt.figure(figsize=(7, 5))
        plt.bar(x - width/2, evo_scores, width, label=method_name)
        plt.bar(x + width/2, baseline_scores, width, label='Baseline')
        plt.xticks(x, metric_labels)
        plt.ylabel("Score")
        plt.title(f"Metric Comparison - {method_name} vs Baseline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_solution_frontier(self, solutions, filepath):
        """
        Plot the frontier of non-dominated solutions.
        """
        if not solutions:
            secure_print(f"Warning: Cannot create frontier plot for {filepath} - empty solutions")
            return

        criterion1_vals = [s.objectives[0] for s in solutions]
        criterion2_vals = [s.objectives[1] for s in solutions]

        plt.figure(figsize=(6, 5))
        plt.scatter(criterion1_vals, criterion2_vals, c='blue', alpha=0.7)
        plt.xlabel("Criterion 1 (minimize)")
        plt.ylabel("Criterion 2 (minimize)")
        plt.title("Solution Frontier (aggregated runs)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def _plot_distribution(self, values, filepath, title="Distribution"):
        """
        Create a boxplot for a list of numeric values.
        """
        if not values:
            secure_print(f"Warning: Cannot create distribution plot for {filepath} - empty values")
            return

        plt.figure()
        plt.boxplot(values)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    ###################### CSV Export Methods ######################

    def _save_frontier_to_csv(self, solutions, filepath):
        """
        Save the frontier solutions to a CSV file.
        """
        if not solutions:
            secure_print(f"Warning: Cannot save empty solution frontier to {filepath}")
            return

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Criterion1', 'Criterion2'])
            for s in solutions:
                writer.writerow(s.objectives)

    def _save_best_candidates_to_csv(self, best_candidates, filepath):
        """
        Save the best candidates from each run to a CSV file.
        """
        if not best_candidates:
            secure_print(f"Warning: Cannot save empty best candidates to {filepath}")
            return

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'Criterion1', 'Criterion2', 'QualityScore'])
            for idx, (sol, quality) in enumerate(best_candidates):
                writer.writerow([idx + 1, sol.objectives[0], sol.objectives[1], quality])


#####################################################
#             Experiment Execution                  #
#####################################################

def create_optimizer(strategy_label, profile, task, max_iterations):
    """
    Create and configure an optimization strategy based on label and profile.
    """
    if strategy_label == 'nsga2':
        base_mutation_rate = 1.0 / task.number_of_variables
        mutation_rate = base_mutation_rate * profile['mutation_intensity']

        crossover = SBXCrossover(probability=profile['crossover_rate'], distribution_index=20.0)
        mutation = PolynomialMutation(probability=mutation_rate, distribution_index=20.0)

        return NSGAII(
            problem=task,
            population_size=profile['population_size'],
            offspring_population_size=profile['offspring_size'],
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_iterations)
        )
    else:
        # Strategy B
        base_mutation_rate = 1.0 / task.number_of_variables
        mutation_rate = base_mutation_rate * profile['mutation_intensity']

        crossover = SBXCrossover(probability=profile['crossover_rate'], distribution_index=20.0)
        mutation = PolynomialMutation(probability=mutation_rate, distribution_index=20.0)

        # Ensure offspring size doesn't exceed population for this strategy
        safe_offspring_size = min(profile['offspring_size'], profile['population_size'])

        return SPEA2(
            problem=task,
            population_size=profile['population_size'],
            offspring_population_size=safe_offspring_size,
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_iterations)
        )

def execute_single_experiment(matrix, strategy_label, profile_label, profile, run_number, max_iterations=4000):
    """
    Execute a single experiment with specific strategy/profile combination.
    Returns a dictionary with results.
    """
    experiment_id = f"{strategy_label}-{profile_label}-run{run_number}"
    secure_print(f"🚀 Starting {experiment_id}")

    try:
        task = OptimizationTaskDualCriteria(matrix, group_count=3)
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

def run_baseline_reference(matrix, n_runs=10):
    """
    Run baseline reference method multiple times and compute various metrics.
    Returns dictionary with statistical results.
    """
    group_count = 3
    compactness_scores = []
    topology_scores = []
    silhouette_scores = []

    best_baseline = None
    best_silhouette = -1
    best_compactness = float('inf')
    best_topology = float('inf')

    for i in range(n_runs):
        baseline = KMeans(n_clusters=group_count, random_state=i)
        baseline.fit(matrix)
        assignments = baseline.labels_

        compactness = np.sum(np.min(pairwise_distances(matrix, baseline.cluster_centers_), axis=1) ** 2)
        topology = AnalysisMetrics.compute_topology_score(matrix, assignments)
        silhouette = silhouette_score(matrix, assignments)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_baseline = {
                'model': baseline,
                'assignments': assignments,
                'compactness': compactness,
                'topology': topology,
                'silhouette': silhouette
            }

        if compactness < best_compactness:
            best_compactness = compactness

        if topology < best_topology:
            best_topology = topology

        compactness_scores.append(compactness)
        topology_scores.append(topology)
        silhouette_scores.append(silhouette)

    compactness_mean = np.mean(compactness_scores)
    compactness_std = np.std(compactness_scores)
    topology_mean = np.mean(topology_scores)
    topology_std = np.std(topology_scores)
    silhouette_mean = np.mean(silhouette_scores)
    silhouette_std = np.std(silhouette_scores)

    secure_print("Baseline reference results:")
    secure_print(f"  Compactness: {compactness_mean:.4f} ± {compactness_std:.4f} (best: {best_compactness:.4f})")
    secure_print(f"  Topology: {topology_mean:.4f} ± {topology_std:.4f} (best: {best_topology:.4f})")
    secure_print(f"  Silhouette: {silhouette_mean:.4f} ± {silhouette_std:.4f} (best: {best_silhouette:.4f})")

    return {
        'compactness': compactness_scores,
        'topology': topology_scores,
        'silhouette': silhouette_scores,
        'best_model': best_baseline,
        'best_compactness': best_compactness,
        'best_topology': best_topology,
        'best_silhouette': best_silhouette,
        'mean_compactness': compactness_mean,
        'mean_topology': topology_mean,
        'mean_silhouette': silhouette_mean,
        'std_compactness': compactness_std,
        'std_topology': topology_std,
        'std_silhouette': silhouette_std
    }

def execute_parallel_experiments(matrix, n_runs=10, max_workers=None):
    """
    Execute all configured experiments in parallel.
    Each experiment runs in a separate thread.
    """
    all_experiments = []
    for strategy_label, (_, profile_dict) in OPTIMIZER_REGISTRY.items():
        for profile_label, profile in profile_dict.items():
            for run_number in range(1, n_runs + 1):
                all_experiments.append((strategy_label, profile_label, profile, run_number))

    total_experiments = len(all_experiments)
    secure_print(f"Preparing to execute {total_experiments} experiments with {max_workers} worker threads")

    output_manager = OutputManager()
    completed = 0

    execute_func = partial(execute_single_experiment, matrix)

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