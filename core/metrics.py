"""
Utility methods for pattern analysis metrics and operations.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from jmetal.core.quality_indicator import HyperVolume


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