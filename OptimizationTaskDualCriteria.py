import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


class OptimizationTaskDualCriteria(Problem[FloatSolution]):
    """
    Twin-criterion optimization task handler for pattern discovery framework.
    Balances internal cohesion and topological preservation metrics.
    """

    def __init__(self, matrix: np.ndarray, group_count: int):
        """
        Initialize optimization parameters for pattern discovery.

        Args:
            matrix: Data elements matrix with shape (elements_count, attributes_count)
            group_count: Number of pattern groups to discover
        """
        super().__init__()

        # Store source data
        self.matrix = matrix
        self.elements_count, self.attributes_count = matrix.shape
        self.group_count = group_count

        # Define optimization dimensions
        self._dimension_count = self.group_count * self.attributes_count
        self._criterion_count = 2
        self._constraint_count = 0

        # Set boundaries for pattern prototype coordinates
        self._lower_limits = []
        self._upper_limits = []
        for i in range(self._dimension_count):
            attribute_idx = i % self.attributes_count
            self._lower_limits.append(float(np.min(matrix[:, attribute_idx])))
            self._upper_limits.append(float(np.max(matrix[:, attribute_idx])))

    def number_of_variables(self) -> int:
        return self._dimension_count

    def number_of_objectives(self) -> int:
        return self._criterion_count

    def number_of_constraints(self) -> int:
        return self._constraint_count

    def name(self) -> str:
        return "OptimizationTaskDualCriteria"

    def create_solution(self) -> FloatSolution:
        """
        Generate a random initial pattern prototype configuration.

        Returns:
            A new random solution instance.
        """
        candidate = FloatSolution(
            self._lower_limits,
            self._upper_limits,
            self._criterion_count,
            self._constraint_count
        )

        # Initialize prototype coordinates within valid ranges
        candidate.variables = [
            np.random.uniform(self._lower_limits[i], self._upper_limits[i])
            for i in range(self._dimension_count)
        ]

        return candidate

    def evaluate(self, candidate: FloatSolution) -> FloatSolution:
        """
        Assess a candidate solution against twin optimization criteria:
            1) Internal cohesion (compactness)
            2) Topological preservation (neighborhood integrity)

        Args:
            candidate: Candidate pattern prototypes encoded as variables.

        Returns:
            The solution with updated criterion scores.
        """
        # Transform candidate variables into pattern prototypes
        prototypes = self._transform_to_prototypes(candidate.variables)

        # Associate elements with nearest prototypes
        assignments = self._assign_elements_to_prototypes(prototypes)

        # Calculate both criterion scores
        cohesion_score = self._calculate_cohesion(assignments, prototypes)
        topology_score = self._calculate_topology_preservation(assignments)

        # Update solution criteria scores
        candidate.objectives[0] = cohesion_score
        candidate.objectives[1] = topology_score

        return candidate

    def _transform_to_prototypes(self, variables: list) -> np.ndarray:
        """
        Reshape flat variables list into prototype matrix.
        """
        return np.array(variables).reshape(self.group_count, self.attributes_count)

    def _assign_elements_to_prototypes(self, prototypes: np.ndarray) -> np.ndarray:
        """
        Determine optimal element-prototype associations.
        """
        distances = pairwise_distances(self.matrix, prototypes, metric='euclidean')
        return np.argmin(distances, axis=1)

    def _calculate_cohesion(self, assignments: np.ndarray, prototypes: np.ndarray) -> float:
        """
        Compute internal cohesion metric (lower is better).

        Args:
            assignments: Group assignment for each element.
            prototypes: Group representative coordinates.

        Returns:
            Total distance between elements and their assigned prototypes.
        """
        total_score = 0.0
        # Calculate distances between all elements and all prototypes
        distances = pairwise_distances(self.matrix, prototypes, metric='euclidean')

        # Sum distances between elements and their assigned prototypes
        for i in range(self.group_count):
            group_elements = (assignments == i)
            total_score += np.sum(distances[group_elements, i])

        return total_score

    def _calculate_topology_preservation(self, assignments: np.ndarray, neighbor_count: int = 10) -> float:
        """
        Compute topology preservation metric based on local neighborhoods.

        Args:
            assignments: Group assignment for each element.
            neighbor_count: Number of nearest neighbors to consider.

        Returns:
            Topology disruption score (lower is better).
        """
        element_count = self.elements_count

        # Find nearest neighbors for each element
        max_neighbors = min(neighbor_count + 1, element_count)
        nn_finder = NearestNeighbors(n_neighbors=max_neighbors, metric='euclidean')
        nn_finder.fit(self.matrix)

        # Remove self-reference (neighbor index 0)
        neighbors = nn_finder.kneighbors(self.matrix, return_distance=False)[:, 1:]

        # Calculate topology disruption
        disruption = 0.0
        for i in range(element_count):
            for j, neighbor_idx in enumerate(neighbors[i]):
                if assignments[i] != assignments[neighbor_idx]:
                    disruption += 1.0 / (j + 1)

        return disruption