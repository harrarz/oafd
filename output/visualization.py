"""
Visualization functions for experiment results.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

from core.metrics import AnalysisMetrics
from core.utils import secure_print


def visualize_groups(task, solution, save_path):
    """
    Create a 2D projection of the grouping solution using PCA.
    
    Args:
        task: Optimization task instance
        solution: Solution object with prototype variables
        save_path: Output file path for the visualization
        
    Returns:
        Boolean indicating success
    """
    if task is None or solution is None:
        secure_print(f"Warning: Cannot create visualization for {save_path} - missing data")
        return False

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
    return True


def compare_with_baseline(matrix, prototypes, method_name, output_path):
    """
    Compare evolutionary solution with baseline (KMeans) on multiple metrics.
    
    Args:
        matrix: Data matrix of shape (elements_count, attributes_count)
        prototypes: Prototype coordinates from optimization
        method_name: Name of the method for labeling
        output_path: Output file path for the comparison chart
        
    Returns:
        Boolean indicating success
    """
    if prototypes is None:
        secure_print(f"Warning: Cannot create comparison for {output_path} - missing prototypes")
        return False

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
    return True


def plot_solution_frontier(solutions, filepath):
    """
    Plot the frontier of non-dominated solutions.
    
    Args:
        solutions: List of solution objects
        filepath: Output file path for the plot
        
    Returns:
        Boolean indicating success
    """
    if not solutions:
        secure_print(f"Warning: Cannot create frontier plot for {filepath} - empty solutions")
        return False

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
    return True


def plot_distribution(values, filepath, title="Distribution"):
    """
    Create a boxplot for a list of numeric values.
    
    Args:
        values: List of numeric values to plot
        filepath: Output file path for the boxplot
        title: Plot title
        
    Returns:
        Boolean indicating success
    """
    if not values:
        secure_print(f"Warning: Cannot create distribution plot for {filepath} - empty values")
        return False

    plt.figure()
    plt.boxplot(values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return True