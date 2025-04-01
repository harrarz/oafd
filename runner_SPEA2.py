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

from ClusteringProblemMultiObjective import ClusteringProblemMultiObjective

# Désactive les logs verbeux de jMetal
logging.getLogger('jmetal').setLevel(logging.WARNING)

#####################################################
#             Global lock & Print utility           #
#####################################################

global_lock = threading.Lock()

def log_print(*args, **kwargs):
    """
    Thread-safe print function with a timestamp.
    Usage: log_print("Message", key=value, ...)
    """
    with global_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}]", *args, **kwargs)

#####################################################
#      Configurations & Algorithm Registry          #
#####################################################

# Configurations élargies pour NSGA-II
NSGA2_CONFIGS = {
    'default': {
        'population_size': 80,
        'offspring_population_size': 40,
        'mutation_prob_multiplier': 1.0,
        'crossover_prob': 0.9
    },
    'explorative': {
        'population_size': 100,
        'offspring_population_size': 80,
        'mutation_prob_multiplier': 2.0,
        'crossover_prob': 0.8
    },
    'exploitative': {
        'population_size': 80,
        'offspring_population_size': 20,
        'mutation_prob_multiplier': 0.5,
        'crossover_prob': 0.95
    },
    'high_mut': {
        'population_size': 80,
        'offspring_population_size': 40,
        'mutation_prob_multiplier': 3.0,
        'crossover_prob': 0.9
    }
}

# Configurations élargies pour SPEA2
SPEA2_CONFIGS = {
    'default': {
        'population_size': 80,
        'offspring_population_size': 40,
        'mutation_prob_multiplier': 1.0,
        'crossover_prob': 0.9
    },
    'large_pop': {
        'population_size': 120,
        'offspring_population_size': 60,
        'mutation_prob_multiplier': 1.0,
        'crossover_prob': 0.9
    },
    'explorative': {
        'population_size': 100,
        'offspring_population_size': 80,
        'mutation_prob_multiplier': 2.0,
        'crossover_prob': 0.8
    },
    'exploitative': {
        'population_size': 80,
        'offspring_population_size': 40,
        'mutation_prob_multiplier': 0.5,
        'crossover_prob': 0.95
    }
}

ALGORITHM_REGISTRY = {
    'nsga2': (NSGAII, NSGA2_CONFIGS),
    'spea2': (SPEA2, SPEA2_CONFIGS)
}

#####################################################
#            Helper Classes & Functions            #
#####################################################

class ClusteringMetrics:
    """
    Utility class for clustering metrics and operations.
    Provides methods to decode solutions, assign clusters, compute connectivity, etc.
    """

    @staticmethod
    def decode_solution(problem, solution):
        """
        Convert a FloatSolution to cluster centroids.

        Args:
            problem: The ClusteringProblemMultiObjective instance.
            solution: FloatSolution containing the flattened centroid coordinates.

        Returns:
            A NumPy array of shape (n_clusters, n_features) for the centroids.
        """
        n_clusters = problem.n_clusters
        n_features = problem.X.shape[1]
        return np.array(solution.variables).reshape(n_clusters, n_features)

    @staticmethod
    def assign_clusters(X, centers):
        """
        Assign data points to their nearest centroids.

        Args:
            X: Input data of shape (n_samples, n_features).
            centers: Centroid coordinates of shape (n_clusters, n_features).

        Returns:
            A NumPy array of labels (cluster indices).
        """
        distances = np.zeros((X.shape[0], centers.shape[0]))
        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        return np.argmin(distances, axis=1)

    @staticmethod
    def compute_connectivity(X, labels, L=10):
        """
        Calculate connectivity measure based on nearest neighbors.

        Args:
            X: Input data of shape (n_samples, n_features).
            labels: Cluster labels assigned to each data point.
            L: Number of nearest neighbors to consider.

        Returns:
            The connectivity value (lower is better).
        """
        N = X.shape[0]
        neigh = NearestNeighbors(n_neighbors=L + 1, metric='euclidean')
        neigh.fit(X)
        neighbors = neigh.kneighbors(X, return_distance=False)[:, 1:]

        conn_value = 0.0
        for i in range(N):
            # Pénalités décroissantes pour les voisins dont le label diffère
            penalties = 1.0 / np.arange(1, L + 1)
            different_label_mask = labels[i] != labels[neighbors[i]]
            conn_value += np.sum(penalties[different_label_mask])
        return conn_value

    @staticmethod
    def compute_hypervolume(solutions):
        """
        Calculate the hypervolume for a set of solutions.

        Args:
            solutions: A list of FloatSolution objects.

        Returns:
            Hypervolume value (float).
        """
        if not solutions:
            return 0.0
        front_objectives = [s.objectives for s in solutions]

        # Détermine un point de référence légèrement plus mauvais
        wcss_max = max(obj[0] for obj in front_objectives)
        conn_max = max(obj[1] for obj in front_objectives)
        reference_point = [wcss_max * 1.05, conn_max * 1.05]

        hv = HyperVolume(reference_point)
        return hv.compute(front_objectives)


class ConvergenceObserver:
    """
    Observer to track the convergence of an algorithm by recording
    evaluations, hypervolumes, mean WCSS, and mean connectivity.
    """

    def __init__(self):
        self.evaluations = []
        self.hypervolumes = []
        self.wcss = []
        self.connectivity = []

    def update(self, **kwargs):
        eval_count = kwargs.get("EVALUATIONS", None)
        solutions = kwargs.get("SOLUTIONS", [])
        max_evals = kwargs.get("MAX_EVALUATIONS", 0)

        if eval_count is not None and solutions:
            # On stocke toutes les 5 évaluations (ou à la dernière évaluation)
            if eval_count % 5 == 0 or eval_count == max_evals:
                hv = ClusteringMetrics.compute_hypervolume(solutions)
                wcss_mean = np.mean([s.objectives[0] for s in solutions])
                conn_mean = np.mean([s.objectives[1] for s in solutions])

                self.evaluations.append(eval_count)
                self.hypervolumes.append(hv)
                self.wcss.append(wcss_mean)
                self.connectivity.append(conn_mean)


class ResultManager:
    """
    Centralized result management class. Collects results from runs,
    manages in-memory data, and performs batch saving of visualizations and CSVs.
    """

    def __init__(self):
        self.results = {}
        self.visualizations_to_save = []

    def add_result(self, result):
        """
        Register a single run's result.

        Args:
            result: Dictionary containing keys such as 'algorithm', 'config',
                    'front', 'hypervolume', 'best_solution', etc.
        """
        algo = result['algorithm']
        config = result['config']

        with global_lock:
            if (algo, config) not in self.results:
                self.results[(algo, config)] = {
                    'all_fronts': [],
                    'hypervolumes': [],
                    'best_runs': [],
                    'best_solution': None,
                    'best_hv': -np.inf,
                    'best_problem': None
                }

            r = self.results[(algo, config)]
            r['all_fronts'].extend(result['front'])
            r['hypervolumes'].append(result['hypervolume'])
            r['best_runs'].append((result['best_solution'], result['best_solution_hv']))

            if result['best_solution_hv'] > r['best_hv']:
                r['best_hv'] = result['best_solution_hv']
                r['best_solution'] = result['best_solution']
                r['best_problem'] = result['problem']

    def save_results(self, X):
        """
        Save all results (Pareto fronts, best solutions, visualizations, etc.)
        in a single batch operation.

        Args:
            X: The original dataset used for clustering.
        """
        with global_lock:
            for (algo, config) in self.results.keys():
                os.makedirs(f"results/{algo}/{config}", exist_ok=True)

        # Préparation des visualisations
        for (algo, config), result in self.results.items():
            results_dir = f"results/{algo}/{config}"

            best_solution = result['best_solution']
            best_problem = result['best_problem']

            # 1. Visualisations programmées
            if best_solution is not None and best_problem is not None:
                # Clustering
                self.visualizations_to_save.append(
                    (self._visualize_clustering,
                     (best_problem, best_solution,
                      os.path.join(results_dir, "best_clustering.png")))
                )
                # Comparaison avec KMeans
                centers = ClusteringMetrics.decode_solution(best_problem, best_solution)
                self.visualizations_to_save.append(
                    (self._compare_with_kmeans,
                     (X, centers, f"{algo.upper()} - {config}",
                      os.path.join(results_dir, "comparative_plot.png")))
                )

            # Pareto front
            self.visualizations_to_save.append(
                (self._plot_pareto_front,
                 (result['all_fronts'], os.path.join(results_dir, "pareto_front.png")))
            )

            # Hypervolume boxplot
            self.visualizations_to_save.append(
                (self._plot_boxplot,
                 (result['hypervolumes'],
                  os.path.join(results_dir, "hypervolume_boxplot.png"),
                  "Hypervolume Distribution"))
            )

            # 2. Sauvegarde des CSV
            self._save_pareto_to_csv(
                result['all_fronts'],
                os.path.join(results_dir, "pareto_front.csv")
            )
            self._save_best_solutions_to_csv(
                result['best_runs'],
                os.path.join(results_dir, "best_solutions.csv")
            )

            log_print(f"✅ Results prepared for {algo} - {config}")

        log_print("Creating visualizations...")
        self.save_batch_visualizations()
        log_print("All visualizations saved!")

    def save_batch_visualizations(self):
        """Process all queued visualizations in a single batch."""
        for viz_func, args in self.visualizations_to_save:
            viz_func(*args)
        self.visualizations_to_save = []

    ###################### Internal Plotting Methods ######################

    def _visualize_clustering(self, problem, solution, save_path):
        """
        Create a 2D visualization of the clustering solution using PCA.
        """
        if problem is None or solution is None:
            log_print(f"Warning: Cannot create visualization for {save_path} - missing data")
            return

        centers = ClusteringMetrics.decode_solution(problem, solution)
        labels = ClusteringMetrics.assign_clusters(problem.X, centers)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(problem.X)
        centers_pca = pca.transform(centers)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=150,
                    marker='x', label='Centroids')
        plt.title("Best Clustering Solution (Hypervolume)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _compare_with_kmeans(self, X, best_centers, method_name, output_path):
        """
        Compare an evolutionary algorithm solution with KMeans on WCSS,
        Connectivity, and Silhouette.
        """
        if best_centers is None:
            log_print(f"Warning: Cannot create comparison for {output_path} - missing centers")
            return

        labels_ea = ClusteringMetrics.assign_clusters(X, best_centers)

        # KMeans
        kmeans = KMeans(n_clusters=len(best_centers), random_state=42).fit(X)
        labels_kmeans = kmeans.labels_

        scores = {
            'Silhouette': [
                silhouette_score(X, labels_ea),
                silhouette_score(X, labels_kmeans)
            ],
            'WCSS': [
                np.sum(np.min(pairwise_distances(X, best_centers), axis=1) ** 2),
                np.sum(np.min(pairwise_distances(X, kmeans.cluster_centers_), axis=1) ** 2)
            ],
            'Connectivity': [
                ClusteringMetrics.compute_connectivity(X, labels_ea),
                ClusteringMetrics.compute_connectivity(X, labels_kmeans)
            ]
        }

        labels_list = list(scores.keys())
        ea_scores = [scores[k][0] for k in labels_list]
        km_scores = [scores[k][1] for k in labels_list]

        x = np.arange(len(labels_list))
        width = 0.35

        plt.figure(figsize=(7, 5))
        plt.bar(x - width/2, ea_scores, width, label=method_name)
        plt.bar(x + width/2, km_scores, width, label='KMeans')
        plt.xticks(x, labels_list)
        plt.ylabel("Score")
        plt.title(f"Comparison of Metrics - {method_name} vs KMeans")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_pareto_front(self, pareto_front, filepath):
        """
        Plot the Pareto front of solutions.
        """
        if not pareto_front:
            log_print(f"Warning: Cannot create Pareto front for {filepath} - empty front")
            return

        wcss_vals = [s.objectives[0] for s in pareto_front]
        conn_vals = [s.objectives[1] for s in pareto_front]

        plt.figure(figsize=(6, 5))
        plt.scatter(wcss_vals, conn_vals, c='blue', alpha=0.7)
        plt.xlabel("WCSS (minimize)")
        plt.ylabel("Connectivity (minimize)")
        plt.title("Pareto Front (combined runs)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def _plot_boxplot(self, values, filepath, title="Boxplot"):
        """
        Create a boxplot for a list of numeric values (e.g., hypervolume).
        """
        if not values:
            log_print(f"Warning: Cannot create boxplot for {filepath} - empty values")
            return

        plt.figure()
        plt.boxplot(values)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    ###################### Internal CSV Export Methods ######################

    def _save_pareto_to_csv(self, pareto_front, filepath):
        """
        Save the Pareto front solutions to a CSV file.
        """
        if not pareto_front:
            log_print(f"Warning: Cannot save empty Pareto front to {filepath}")
            return

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['WCSS', 'Connectivity'])
            for s in pareto_front:
                writer.writerow(s.objectives)

    def _save_best_solutions_to_csv(self, best_runs, filepath):
        """
        Save the best solutions from each run to a CSV file.
        """
        if not best_runs:
            log_print(f"Warning: Cannot save empty best runs to {filepath}")
            return

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'WCSS', 'Connectivity', 'Hypervolume'])
            for idx, (sol, hv) in enumerate(best_runs):
                writer.writerow([idx + 1, sol.objectives[0], sol.objectives[1], hv])


#####################################################
#             Parallel Execution Methods            #
#####################################################

def create_algorithm(algo_label, config, problem, max_evals):
    """
    Create and configure an optimization algorithm (NSGA-II or SPEA2),
    given its label, configuration, problem, and max evaluations.
    """
    if algo_label == 'nsga2':
        base_mutation_prob = 1.0 / problem.number_of_variables
        mutation_prob = base_mutation_prob * config['mutation_prob_multiplier']

        crossover = SBXCrossover(probability=config['crossover_prob'], distribution_index=20.0)
        mutation = PolynomialMutation(probability=mutation_prob, distribution_index=20.0)

        return NSGAII(
            problem=problem,
            population_size=config['population_size'],
            offspring_population_size=config['offspring_population_size'],
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
        )
    else:
        # SPEA2
        base_mutation_prob = 1.0 / problem.number_of_variables
        mutation_prob = base_mutation_prob * config['mutation_prob_multiplier']

        crossover = SBXCrossover(probability=config['crossover_prob'], distribution_index=20.0)
        mutation = PolynomialMutation(probability=mutation_prob, distribution_index=20.0)

        # Évite un offspring_population_size trop grand pour SPEA2
        safe_offspring_size = min(config['offspring_population_size'], config['population_size'])

        return SPEA2(
            problem=problem,
            population_size=config['population_size'],
            offspring_population_size=safe_offspring_size,
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
        )

def run_single_experiment_run(X, algo_label, config_label, config, run_number, max_evals=4000):
    """
    Run a single experiment (one algorithm/config combination, single run).
    Returns a dictionary with run results.
    """
    run_id = f"{algo_label}-{config_label}-run{run_number}"
    log_print(f"🚀 Starting {run_id}")

    try:
        problem = ClusteringProblemMultiObjective(X, n_clusters=3)
        algorithm = create_algorithm(algo_label, config, problem, max_evals)

        # Simple observer
        observer = ConvergenceObserver()
        algorithm.observable.register(observer)

        algorithm.run()
        front = algorithm.result()

        current_hv = ClusteringMetrics.compute_hypervolume(front)

        # Best individual solution by hypervolume
        best_sol = max(front, key=lambda s: ClusteringMetrics.compute_hypervolume([s]))

        log_print(f"✅ Completed {run_id}")

        return {
            'id': run_id,
            'algorithm': algo_label,
            'config': config_label,
            'run': run_number,
            'front': front,
            'hypervolume': current_hv,
            'best_solution': best_sol,
            'best_solution_hv': ClusteringMetrics.compute_hypervolume([best_sol]),
            'problem': problem
        }
    except Exception as e:
        log_print(f"❌ Error in {run_id}: {str(e)}")
        raise

def run_kmeans_benchmarks(X, n_runs=10):
    """
    Run K-means multiple times and compute WCSS, Connectivity, and Silhouette.
    Returns a dictionary with stats (means, best, std, etc.).
    """
    k = 3
    wcss_scores = []
    connectivity_scores = []
    silhouette_scores = []

    best_kmeans = None
    best_silhouette = -1
    best_wcss = float('inf')
    best_conn = float('inf')

    for i in range(n_runs):
        kmeans = KMeans(n_clusters=k, random_state=i)
        kmeans.fit(X)
        labels = kmeans.labels_

        wcss = np.sum(np.min(pairwise_distances(X, kmeans.cluster_centers_), axis=1) ** 2)
        conn = ClusteringMetrics.compute_connectivity(X, labels)
        sil = silhouette_score(X, labels)

        if sil > best_silhouette:
            best_silhouette = sil
            best_kmeans = {
                'model': kmeans,
                'labels': labels,
                'wcss': wcss,
                'connectivity': conn,
                'silhouette': sil
            }

        if wcss < best_wcss:
            best_wcss = wcss

        if conn < best_conn:
            best_conn = conn

        wcss_scores.append(wcss)
        connectivity_scores.append(conn)
        silhouette_scores.append(sil)

    wcss_mean = np.mean(wcss_scores)
    wcss_std = np.std(wcss_scores)
    conn_mean = np.mean(connectivity_scores)
    conn_std = np.std(connectivity_scores)
    sil_mean = np.mean(silhouette_scores)
    sil_std = np.std(silhouette_scores)

    log_print("K-means benchmark results:")
    log_print(f"  WCSS: {wcss_mean:.4f} ± {wcss_std:.4f} (best: {best_wcss:.4f})")
    log_print(f"  Connectivity: {conn_mean:.4f} ± {conn_std:.4f} (best: {best_conn:.4f})")
    log_print(f"  Silhouette: {sil_mean:.4f} ± {sil_std:.4f} (best: {best_silhouette:.4f})")

    return {
        'wcss': wcss_scores,
        'connectivity': connectivity_scores,
        'silhouette': silhouette_scores,
        'best_model': best_kmeans,
        'best_wcss': best_wcss,
        'best_connectivity': best_conn,
        'best_silhouette': best_silhouette,
        'mean_wcss': wcss_mean,
        'mean_connectivity': conn_mean,
        'mean_silhouette': sil_mean,
        'std_wcss': wcss_std,
        'std_connectivity': conn_std,
        'std_silhouette': sil_std
    }

def run_experiments_parallel(X, n_runs=10, max_workers=None):
    """
    Run all configured experiments (NSGA2 + SPEA2 + various configs) in parallel.
    Each run is executed on a separate thread.
    """
    all_runs = []
    for algo_label, (_, config_dict) in ALGORITHM_REGISTRY.items():
        for config_label, config in config_dict.items():
            for run_number in range(1, n_runs + 1):
                all_runs.append((algo_label, config_label, config, run_number))

    total_runs = len(all_runs)
    log_print(f"Preparing to run {total_runs} individual experiment runs with {max_workers} worker threads")

    result_manager = ResultManager()
    completed = 0

    run_func = partial(run_single_experiment_run, X)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_func, algo, cfg_label, cfg, r_num)
            for algo, cfg_label, cfg, r_num in all_runs
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                result_manager.add_result(result)
                completed += 1
                if completed % 5 == 0 or completed == total_runs:
                    log_print(f"Progress: {completed}/{total_runs} runs complete ({completed / total_runs * 100:.1f}%)")
            except Exception as e:
                log_print(f"❌ Error in experiment: {e}")

    log_print("All runs completed, saving final results...")
    result_manager.save_results(X)
    log_print("✅ All experiments completed!")

#####################################################
#                     Main script                   #
#####################################################

def main():
    """
    Main function to load datasets, run benchmarks (K-means) and
    evolutionary algorithms (NSGA-II, SPEA2), then save results to disk.
    """
    datasets = {
        'iris': load_iris().data,
        # Ajoutez d’autres jeux de données si nécessaire
    }

    cpu_count = multiprocessing.cpu_count()
    thread_count = min(cpu_count + 2, 12)

    all_results = {}

    os.makedirs("results/summary", exist_ok=True)
    with open("results/summary/algorithm_comparison.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Dataset', 'Algorithm', 'Config',
            'WCSS_Mean', 'WCSS_Std', 'WCSS_Best',
            'Connectivity_Mean', 'Connectivity_Std', 'Connectivity_Best',
            'Silhouette_Mean', 'Silhouette_Std', 'Silhouette_Best',
            'Hypervolume_Mean', 'Hypervolume_Std'
        ])

    for dataset_name, X in datasets.items():
        log_print(f"=== Starting experiments on {dataset_name} dataset ===")

        log_print(f"Running K-means benchmarks on {dataset_name}...")
        kmeans_results = run_kmeans_benchmarks(X, n_runs=10)
        all_results[f'kmeans_{dataset_name}'] = kmeans_results

        log_print(f"Running evolutionary algorithms on {dataset_name}...")
        run_experiments_parallel(X, n_runs=10, max_workers=thread_count)

        # On marque juste la fin de l’exécution pour ce dataset
        all_results[f'mo_{dataset_name}'] = {'processed': True}

        # Sauvegarde de K-means uniquement
        with open("results/summary/algorithm_comparison.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                dataset_name, 'K-means', 'N/A',
                kmeans_results['mean_wcss'], kmeans_results['std_wcss'], kmeans_results['best_wcss'],
                kmeans_results['mean_connectivity'], kmeans_results['std_connectivity'], kmeans_results['best_connectivity'],
                kmeans_results['mean_silhouette'], kmeans_results['std_silhouette'], kmeans_results['best_silhouette'],
                'N/A', 'N/A'
            ])

    log_print("Experiments completed successfully. Results saved to disk.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    start_time = datetime.now()
    log_print(f"Starting clustering experiments at {start_time}")

    main()

    end_time = datetime.now()
    duration = end_time - start_time
    log_print(f"Experiments completed in {duration.total_seconds() / 60:.2f} minutes")