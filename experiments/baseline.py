"""
Output management and result collection.
"""
import os
import csv
import threading
import numpy as np

from core.utils import secure_print
from core.metrics import AnalysisMetrics
from output.visualization import (
    visualize_groups, compare_with_baseline, 
    plot_solution_frontier, plot_distribution
)


class OutputManager:
    """
    Manages experiment output collection, visualization, and persistence.
    """

    def __init__(self):
        """Initialize the output manager."""
        self.collected_results = {}
        self.visualizations_queue = []
        self.sync_lock = threading.Lock()

    def add_result(self, result):
        """
        Register a single experiment result.

        Args:
            result: Dictionary containing experiment outcome data.
        """
        strategy = result['strategy']
        profile = result['profile']

        with self.sync_lock:
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
        with self.sync_lock:
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
                    (visualize_groups,
                     (source_task, top_solution,
                      os.path.join(output_dir, "top_grouping.png")))
                )
                # Compare with baseline method
                prototypes = AnalysisMetrics.transform_solution(source_task, top_solution)
                self.visualizations_queue.append(
                    (compare_with_baseline,
                     (matrix, prototypes, f"{strategy.upper()} - {profile}",
                      os.path.join(output_dir, "comparison_plot.png")))
                )

            # Solution frontier
            self.visualizations_queue.append(
                (plot_solution_frontier,
                 (result['all_solutions'], os.path.join(output_dir, "solution_frontier.png")))
            )

            # Quality score distribution
            self.visualizations_queue.append(
                (plot_distribution,
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

    ###################### CSV Export Methods ######################

    def _save_frontier_to_csv(self, solutions, filepath):
        """
        Save the frontier solutions to a CSV file.
        
        Args:
            solutions: List of solution objects
            filepath: Output CSV file path
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
        
        Args:
            best_candidates: List of tuples (solution, quality_score)
            filepath: Output CSV file path
        """
        if not best_candidates:
            secure_print(f"Warning: Cannot save empty best candidates to {filepath}")
            return

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'Criterion1', 'Criterion2', 'QualityScore'])
            for idx, (sol, quality) in enumerate(best_candidates):
                writer.writerow([idx + 1, sol.objectives[0], sol.objectives[1], quality])