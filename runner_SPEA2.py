import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import logging
import threading
import concurrent.futures
from time import sleep
from datetime import datetime
import multiprocessing
from functools import partial
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

# Disable the verbose jMetal logging
logging.getLogger('jmetal').setLevel(logging.WARNING)

# Global lock for all operations - simpler but more efficient
global_lock = threading.Lock()

#####################################################
# Configuration definitions for different algorithms #
#####################################################

# Expanded NSGA-II configurations for direct comparison
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

# Expanded SPEA2 configurations for direct comparison
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
    # 'small_pop' removed as it causes errors
    'exploitative': {  
        'population_size': 80,
        'offspring_population_size': 40,  # Match parent population for SPEA2
        'mutation_prob_multiplier': 0.5,
        'crossover_prob': 0.95
    }
}

# Algorithm registry
ALGORITHM_REGISTRY = {
    'nsga2': (NSGAII, NSGA2_CONFIGS),
    'spea2': (SPEA2, SPEA2_CONFIGS)
}

# Simplified print function
def log_print(*args, **kwargs):
    """Thread-safe print function with timestamp"""
    with global_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}]", *args, **kwargs)

#####################################################
#           Helper classes and functions            #
#####################################################

class ClusteringMetrics:
    """Utility class for clustering metrics and operations"""
    
    @staticmethod
    def decode_solution(problem, solution):
        """Convert a solution to cluster centroids"""
        n_clusters = problem.n_clusters
        n_features = problem.X.shape[1]
        return np.array(solution.variables).reshape(n_clusters, n_features)
    
    @staticmethod
    def assign_clusters(X, centers):
        """Assign data points to nearest centroids"""
        distances = np.zeros((X.shape[0], centers.shape[0]))
        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        return np.argmin(distances, axis=1)
    
    @staticmethod
    def compute_connectivity(X, labels, L=10):
        """Calculate connectivity measure - optimized implementation"""
        N = X.shape[0]
        # Use sklearn for efficient neighbor computation
        neigh = NearestNeighbors(n_neighbors=L+1, metric='euclidean')
        neigh.fit(X)
        neighbors = neigh.kneighbors(X, return_distance=False)[:, 1:]
        
        # Vectorized computation
        conn_value = 0.0
        for i in range(N):
            # Calculate penalty for each neighbor with different label
            penalties = 1.0 / np.arange(1, L + 1)
            different_label_mask = labels[i] != labels[neighbors[i]]
            conn_value += np.sum(penalties[different_label_mask])
            
        return conn_value
    
    @staticmethod
    def compute_hypervolume(solutions):
        """Calculate hypervolume indicator for a set of solutions"""
        if not solutions:
            return 0.0
            
        front_objectives = [s.objectives for s in solutions]
        wcss_max = max(obj[0] for obj in front_objectives)
        conn_max = max(obj[1] for obj in front_objectives)
        
        # Set reference point slightly worse than worst solution
        reference_point = [wcss_max * 1.05, conn_max * 1.05]
        hv = HyperVolume(reference_point)
        return hv.compute(front_objectives)


class ConvergenceObserver:
    """Simplified observer to track the convergence of an algorithm"""
    
    def __init__(self):
        self.evaluations = []
        self.hypervolumes = []
        self.wcss = []
        self.connectivity = []
    
    def update(self, **kwargs):
        eval_count = kwargs.get("EVALUATIONS", None)
        solutions = kwargs.get("SOLUTIONS", [])
        
        if eval_count is not None and solutions:
            # Only track every 5th evaluation to reduce overhead
            if eval_count % 5 == 0 or eval_count == kwargs.get("MAX_EVALUATIONS", 0):
                hv = ClusteringMetrics.compute_hypervolume(solutions)
                wcss_mean = np.mean([s.objectives[0] for s in solutions])
                conn_mean = np.mean([s.objectives[1] for s in solutions])
                
                self.evaluations.append(eval_count)
                self.hypervolumes.append(hv)
                self.wcss.append(wcss_mean)
                self.connectivity.append(conn_mean)


class ResultManager:
    """Class for centralized result management - with improved efficiency"""
    
    def __init__(self):
        self.results = {}
        # In-memory cache of results to reduce file I/O
        self.visualizations_to_save = []
    
    def add_result(self, result):
        """Add a result to the collection - minimal locking"""
        algo = result['algorithm']
        config = result['config']
        
        with global_lock:
            # Initialize this algorithm/config if needed
            if (algo, config) not in self.results:
                self.results[(algo, config)] = {
                    'all_fronts': [],
                    'hypervolumes': [],
                    'best_runs': [],
                    'best_solution': None,
                    'best_hv': -np.inf,
                    'best_problem': None
                }
            
            # Add to the results
            r = self.results[(algo, config)]
            r['all_fronts'].extend(result['front'])
            r['hypervolumes'].append(result['hypervolume'])
            r['best_runs'].append((result['best_solution'], result['best_solution_hv']))
            
            # Update best solution if better
            if result['best_solution_hv'] > r['best_hv']:
                r['best_hv'] = result['best_solution_hv']
                r['best_solution'] = result['best_solution']
                r['best_problem'] = result['problem']
    
    def _save_visualization(self, viz_func, *args):
        """Save all visualizations in a batch"""
        viz_func(*args)
    
    def save_batch_visualizations(self):
        """Process all visualizations in a single batch"""
        # Create all visualization figures in one batch to reduce overhead
        for viz_func, args in self.visualizations_to_save:
            self._save_visualization(viz_func, *args)
        
        # Clear the queue
        self.visualizations_to_save = []
    
    def save_results(self, X):
        """Save all results to files - optimized batch processing"""
        with global_lock:
            # First, create all necessary directories
            for (algo, config) in self.results.keys():
                os.makedirs(f"results/{algo}/{config}", exist_ok=True)
        
        # Prepare all visualization tasks
        for (algo, config), result in self.results.items():
            results_dir = f"results/{algo}/{config}"
            
            # 1. Schedule visualization tasks
            best_centers = None
            if result['best_solution'] is not None and result['best_problem'] is not None:
                best_centers = ClusteringMetrics.decode_solution(
                    result['best_problem'], result['best_solution']
                )
                
                # Add clustering visualization to queue
                self.visualizations_to_save.append(
                    (self._visualize_clustering, 
                     (result['best_problem'], result['best_solution'], 
                      os.path.join(results_dir, "best_clustering.png")))
                )
                
                # Add KMeans comparison to queue
                self.visualizations_to_save.append(
                    (self._compare_with_kmeans,
                     (X, best_centers, f"{algo.upper()} - {config}", 
                      os.path.join(results_dir, "comparative_plot.png")))
                )
            
            # Add Pareto front plot to queue
            self.visualizations_to_save.append(
                (self._plot_pareto_front,
                 (result['all_fronts'], os.path.join(results_dir, "pareto_front.png")))
            )
            
            # Add hypervolume boxplot to queue
            self.visualizations_to_save.append(
                (self._plot_boxplot,
                 (result['hypervolumes'], os.path.join(results_dir, "hypervolume_boxplot.png"),
                  "Hypervolume Distribution"))
            )
            
            # 2. Save CSV files (these are small and fast operations)
            self._save_pareto_to_csv(
                result['all_fronts'], 
                os.path.join(results_dir, "pareto_front.csv")
            )
            
            self._save_best_solutions_to_csv(
                result['best_runs'], 
                os.path.join(results_dir, "best_solutions.csv")
            )
            
            log_print(f"✅ Results prepared for {algo} - {config}")
        
        # Process all visualizations in a single batch
        log_print("Creating visualizations...")
        self.save_batch_visualizations()
        log_print("All visualizations saved!")
    
    def _visualize_clustering(self, problem, solution, save_path):
        """Create a 2D visualization of the clustering solution using PCA"""
        if problem is None or solution is None:
            log_print(f"Warning: Cannot create visualization for {save_path} - missing data")
            return
            
        centers = ClusteringMetrics.decode_solution(problem, solution)
        labels = ClusteringMetrics.assign_clusters(problem.X, centers)
        
        # Apply PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(problem.X)
        centers_pca = pca.transform(centers)
        
        # Plot clustered data and centroids
        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=150, marker='x', label='Centroids')
        plt.title("Best Clustering Solution (Hypervolume)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _compare_with_kmeans(self, X, best_centers, method_name, output_path):
        """Compare evolutionary algorithm solution with KMeans"""
        if best_centers is None:
            log_print(f"Warning: Cannot create comparison for {output_path} - missing centers")
            return
            
        labels_ea = ClusteringMetrics.assign_clusters(X, best_centers)
        
        # Run KMeans for comparison
        kmeans = KMeans(n_clusters=len(best_centers), random_state=42).fit(X)
        labels_kmeans = kmeans.labels_
        
        # Calculate metrics for both methods
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
        
        # Prepare bar chart
        labels = list(scores.keys())
        ea_scores = [scores[k][0] for k in labels]
        km_scores = [scores[k][1] for k in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Plot comparison
        plt.figure(figsize=(7, 5))
        plt.bar(x - width/2, ea_scores, width, label=method_name)
        plt.bar(x + width/2, km_scores, width, label='KMeans')
        plt.xticks(x, labels)
        plt.ylabel("Score")
        plt.title(f"Comparison of Metrics - {method_name} vs KMeans")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_pareto_front(self, pareto_front, filepath):
        """Plot the Pareto front of solutions"""
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
        """Create boxplot visualization"""
        if not values:
            log_print(f"Warning: Cannot create boxplot for {filepath} - empty values")
            return
            
        plt.figure()
        plt.boxplot(values)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _save_pareto_to_csv(self, pareto_front, filepath):
        """Save Pareto front solutions to CSV file"""
        if not pareto_front:
            log_print(f"Warning: Cannot save empty Pareto front to {filepath}")
            return
            
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['WCSS', 'Connectivity'])
            for s in pareto_front:
                writer.writerow(s.objectives)
    
    def _save_best_solutions_to_csv(self, best_runs, filepath):
        """Save best solutions from each run to CSV file"""
        if not best_runs:
            log_print(f"Warning: Cannot save empty best runs to {filepath}")
            return
            
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'WCSS', 'Connectivity', 'Hypervolume'])
            for idx, (s, hv) in enumerate(best_runs):
                writer.writerow([idx + 1, s.objectives[0], s.objectives[1], hv])


#####################################################
#            Parallel Execution Framework           #
#####################################################

def create_algorithm(algo_label, config, problem, max_evals):
    """Create and configure an optimization algorithm - optimized for performance"""
    
    if algo_label == 'nsga2':
        # Calculate mutation probability
        base_mutation_prob = 1.0 / problem.number_of_variables
        mutation_prob = base_mutation_prob * config['mutation_prob_multiplier']
        
        # Setup operators
        crossover = SBXCrossover(
            probability=config['crossover_prob'],
            distribution_index=20.0
        )
        mutation = PolynomialMutation(
            probability=mutation_prob,
            distribution_index=20.0
        )
        
        # Create NSGA-II algorithm
        return NSGAII(
            problem=problem,
            population_size=config['population_size'],
            offspring_population_size=config['offspring_population_size'],
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
        )
    else:  # SPEA2
        # Calculate mutation probability
        base_mutation_prob = 1.0 / problem.number_of_variables
        mutation_prob = base_mutation_prob * config['mutation_prob_multiplier']
        
        # Setup operators
        crossover = SBXCrossover(
            probability=config['crossover_prob'],
            distribution_index=20.0
        )
        mutation = PolynomialMutation(
            probability=mutation_prob,
            distribution_index=20.0
        )
        
        # Create SPEA2 algorithm
        # Ensure offspring_population_size is not larger than population_size
        # For SPEA2, we limit the offspring size to avoid "Wrong number of parents" error
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
    """Run a single experiment run (one configuration, one run) - optimized for speed"""
    
    run_id = f"{algo_label}-{config_label}-run{run_number}"
    log_print(f"🚀 Starting {run_id}")
    
    try:
        # Create problem instance for this run
        problem = ClusteringProblemMultiObjective(X, n_clusters=3)
        
        # Create and configure algorithm
        algorithm = create_algorithm(algo_label, config, problem, max_evals)
        
        # Setup observers - lightweight observer
        observer = ConvergenceObserver()
        algorithm.observable.register(observer)
        
        # Run the algorithm
        algorithm.run()
        front = algorithm.result()
        
        # Calculate hypervolume for this run
        current_hv = ClusteringMetrics.compute_hypervolume(front)
        
        # Find best individual solution based on hypervolume contribution
        best_sol = max(front, key=lambda s: ClusteringMetrics.compute_hypervolume([s]))
        
        log_print(f"✅ Completed {run_id}")
        
        # Return the results of this run
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
    """Run multiple K-means runs for comparison"""
    k = 3  # Same number of clusters
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
        
        # Calculate metrics
        wcss = np.sum(np.min(pairwise_distances(X, kmeans.cluster_centers_), axis=1) ** 2)
        conn = ClusteringMetrics.compute_connectivity(X, labels)
        sil = silhouette_score(X, labels)
        
        # Track best solution based on silhouette score (higher is better)
        if sil > best_silhouette:
            best_silhouette = sil
            best_kmeans = {
                'model': kmeans,
                'labels': labels,
                'wcss': wcss,
                'connectivity': conn,
                'silhouette': sil
            }
        
        # Track best WCSS (lower is better)
        if wcss < best_wcss:
            best_wcss = wcss
            
        # Track best connectivity (lower is better)
        if conn < best_conn:
            best_conn = conn
            
        wcss_scores.append(wcss)
        connectivity_scores.append(conn)
        silhouette_scores.append(sil)
    
    # Calculate mean and standard deviation for reporting
    wcss_mean = np.mean(wcss_scores)
    wcss_std = np.std(wcss_scores)
    conn_mean = np.mean(connectivity_scores)
    conn_std = np.std(connectivity_scores)
    sil_mean = np.mean(silhouette_scores)
    sil_std = np.std(silhouette_scores)
    
    log_print(f"K-means benchmark results:")
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
    """Run all experiments in parallel, with individual runs as the unit of parallelism"""
    
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
            executor.submit(run_func, algo, config_label, config, run_num)
            for algo, config_label, config, run_num in all_runs
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                
                result_manager.add_result(result)
                
                completed += 1
                if completed % 5 == 0 or completed == total_runs:
                    log_print(f"Progress: {completed}/{total_runs} runs complete ({completed/total_runs*100:.1f}%)")
                
            except Exception as e:
                log_print(f"❌ Error in experiment: {e}")
    
    log_print("All runs completed, saving final results...")
    result_manager.save_results(X)
    
    log_print("✅ All experiments completed!")
    
def create_comparative_visualizations(all_results):
    """Create visualizations comparing K-means and multi-objective algorithms"""
    
    # Create the comparisons directory
    os.makedirs("results/comparisons", exist_ok=True)
    
    # For each dataset
    for dataset_name in set(k.split('_')[1] for k in all_results.keys()):
        # Get K-means results
        kmeans_key = f'kmeans_{dataset_name}'
        if kmeans_key not in all_results:
            continue
            
        kmeans_data = all_results[kmeans_key]
        
        # Get the best results from multi-objective algorithms
        mo_key = f'mo_{dataset_name}'
        if mo_key not in all_results:
            continue
            
        mo_data = all_results[mo_key]
        
        # Create a comparative boxplot for WCSS
        plt.figure(figsize=(10, 6))
        data_to_plot = [
            kmeans_data['wcss'],
            mo_data['nsga2']['default']['wcss'],
            mo_data['spea2']['default']['wcss']  # Updated to SPEA2
        ]
        plt.boxplot(data_to_plot, labels=['K-means', 'NSGA-II', 'SPEA2'])
        plt.title(f'WCSS Comparison - {dataset_name} dataset')
        plt.ylabel('WCSS (lower is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"results/comparisons/{dataset_name}_wcss_comparison.png")
        plt.close()
        
        # Create a comparative boxplot for Connectivity
        plt.figure(figsize=(10, 6))
        data_to_plot = [
            kmeans_data['connectivity'],
            mo_data['nsga2']['default']['connectivity'],
            mo_data['spea2']['default']['connectivity']
        ]
        plt.boxplot(data_to_plot, labels=['K-means', 'NSGA-II', 'SPEA2'])
        plt.title(f'Connectivity Comparison - {dataset_name} dataset')
        plt.ylabel('Connectivity (lower is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"results/comparisons/{dataset_name}_connectivity_comparison.png")
        plt.close()
        
        # Create a comparative boxplot for Silhouette
        plt.figure(figsize=(10, 6))
        data_to_plot = [
            kmeans_data['silhouette'],
            mo_data['nsga2']['default']['silhouette'],
            mo_data['spea2']['default']['silhouette']
        ]
        plt.boxplot(data_to_plot, labels=['K-means', 'NSGA-II', 'SPEA2'])
        plt.title(f'Silhouette Comparison - {dataset_name} dataset')
        plt.ylabel('Silhouette (higher is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"results/comparisons/{dataset_name}_silhouette_comparison.png")
        plt.close()
        
        # Create a bar chart comparing best solutions
        plt.figure(figsize=(12, 7))
        metrics = ['WCSS', 'Connectivity', 'Silhouette']
        
        # For WCSS and Connectivity, lower is better
        # For Silhouette, higher is better
        kmeans_vals = [
            kmeans_data['best_wcss'] / max(kmeans_data['best_wcss'], 
                                          mo_data['nsga2']['default']['best_wcss'], 
                                          mo_data['spea2']['default']['best_wcss']), 
            kmeans_data['best_connectivity'] / max(kmeans_data['best_connectivity'], 
                                                 mo_data['nsga2']['default']['best_connectivity'], 
                                                 mo_data['spea2']['default']['best_connectivity']),
            kmeans_data['best_silhouette'] / max(kmeans_data['best_silhouette'], 
                                               mo_data['nsga2']['default']['best_silhouette'], 
                                               mo_data['spea2']['default']['best_silhouette'])
        ]
        
        nsga2_vals = [
            mo_data['nsga2']['default']['best_wcss'] / max(kmeans_data['best_wcss'], 
                                                         mo_data['nsga2']['default']['best_wcss'], 
                                                         mo_data['spea2']['default']['best_wcss']),
            mo_data['nsga2']['default']['best_connectivity'] / max(kmeans_data['best_connectivity'], 
                                                                 mo_data['nsga2']['default']['best_connectivity'], 
                                                                 mo_data['spea2']['default']['best_connectivity']),
            mo_data['nsga2']['default']['best_silhouette'] / max(kmeans_data['best_silhouette'], 
                                                               mo_data['nsga2']['default']['best_silhouette'], 
                                                               mo_data['spea2']['default']['best_silhouette'])
        ]
        
        spea2_vals = [
            mo_data['spea2']['default']['best_wcss'] / max(kmeans_data['best_wcss'], 
                                                         mo_data['nsga2']['default']['best_wcss'], 
                                                         mo_data['spea2']['default']['best_wcss']),
            mo_data['spea2']['default']['best_connectivity'] / max(kmeans_data['best_connectivity'], 
                                                                 mo_data['nsga2']['default']['best_connectivity'], 
                                                                 mo_data['spea2']['default']['best_connectivity']),
            mo_data['spea2']['default']['best_silhouette'] / max(kmeans_data['best_silhouette'], 
                                                               mo_data['nsga2']['default']['best_silhouette'], 
                                                               mo_data['spea2']['default']['best_silhouette'])
        ]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width, kmeans_vals, width, label='K-means')
        rects2 = ax.bar(x, nsga2_vals, width, label='NSGA-II')
        rects3 = ax.bar(x + width, spea2_vals, width, label='SPEA2')
        
        ax.set_ylabel('Normalized Score')
        ax.set_title(f'Best Solutions Comparison - {dataset_name} dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value annotations on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        plt.tight_layout()
        plt.savefig(f"results/comparisons/{dataset_name}_best_solutions_comparison.png")
        plt.close()
        
        # Create the Pareto front comparison figure (as requested in the assignment)
        plt.figure(figsize=(10, 8))
        
        # Plot K-means point
        plt.scatter(kmeans_data['best_wcss'], kmeans_data['best_connectivity'], 
                   color='red', s=100, marker='X', label='K-means (best)')
        
        # Plot NSGA-II Pareto front
        nsga2_wcss = [s.objectives[0] for s in mo_data['nsga2']['default']['all_fronts']]
        nsga2_conn = [s.objectives[1] for s in mo_data['nsga2']['default']['all_fronts']]
        plt.scatter(nsga2_wcss, nsga2_conn, color='blue', alpha=0.7, label='NSGA-II Pareto front')
        
        # Plot SPEA2 Pareto front
        spea2_wcss = [s.objectives[0] for s in mo_data['spea2']['default']['all_fronts']]
        spea2_conn = [s.objectives[1] for s in mo_data['spea2']['default']['all_fronts']]
        plt.scatter(spea2_wcss, spea2_conn, color='green', alpha=0.7, label='SPEA2 Pareto front')
        
        # Labels and styles
        plt.xlabel('WCSS (minimize)')
        plt.ylabel('Connectivity (minimize)')
        plt.title(f'Pareto Fronts Comparison - {dataset_name} dataset')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/comparisons/{dataset_name}_pareto_fronts.png")
        plt.close()
        
        # Create hypervolume comparison
        plt.figure(figsize=(8, 6))
        hvs = [
            mo_data['nsga2']['default']['hypervolume'],
            mo_data['spea2']['default']['hypervolume']
        ]
        plt.boxplot(hvs, labels=['NSGA-II', 'SPEA2'])
        plt.title(f'Hypervolume Distribution - {dataset_name} dataset')
        plt.ylabel('Hypervolume (higher is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"results/comparisons/{dataset_name}_hypervolume_comparison.png")
        plt.close()


def main():
    """Main function to run all experiments in parallel"""
    datasets = {
        'iris': load_iris().data,
    }
    
    cpu_count = multiprocessing.cpu_count()
    thread_count = min(cpu_count + 2, 12)
    
    all_results = {}
    
    os.makedirs("results/summary", exist_ok=True)
    with open("results/summary/algorithm_comparison.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset', 'Algorithm', 'Config', 
                         'WCSS_Mean', 'WCSS_Std', 'WCSS_Best',
                         'Connectivity_Mean', 'Connectivity_Std', 'Connectivity_Best',
                         'Silhouette_Mean', 'Silhouette_Std', 'Silhouette_Best',
                         'Hypervolume_Mean', 'Hypervolume_Std'])
    
    for dataset_name, X in datasets.items():
        log_print(f"=== Starting experiments on {dataset_name} dataset ===")
        
        log_print(f"Running K-means benchmarks on {dataset_name}...")
        kmeans_results = run_kmeans_benchmarks(X, n_runs=10)
        
        all_results[f'kmeans_{dataset_name}'] = kmeans_results
        
        log_print(f"Running evolutionary algorithms on {dataset_name}...")
        run_experiments_parallel(X, n_runs=10, max_workers=thread_count)
        
        all_results[f'mo_{dataset_name}'] = {'processed': True}
        
        with open("results/summary/algorithm_comparison.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow([
                dataset_name, 'K-means', 'N/A',
                kmeans_results['mean_wcss'], kmeans_results['std_wcss'], kmeans_results['best_wcss'],
                kmeans_results['mean_connectivity'], kmeans_results['std_connectivity'], kmeans_results['best_connectivity'],
                kmeans_results['mean_silhouette'], kmeans_results['std_silhouette'], kmeans_results['best_silhouette'],
                'N/A', 'N/A'  # K-means doesn't have hypervolume
            ])
    
    log_print("Experiments completed successfully. Results saved to disk.")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    start_time = datetime.now()
    log_print(f"Starting clustering experiments at {start_time}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    log_print(f"Experiments completed in {duration.total_seconds()/60:.2f} minutes")