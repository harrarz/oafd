import numpy as np
from core.metrics import AnalysisMetrics


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
        """
        Observer method to capture algorithm progress.
        
        Args:
            kwargs: Dictionary with keys:
                - EVALUATIONS: Current iteration count
                - SOLUTIONS: Current set of solutions
                - MAX_EVALUATIONS: Maximum iterations
        """
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