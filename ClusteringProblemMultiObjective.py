import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


class ClusteringProblemMultiObjective(Problem[FloatSolution]):
    """
    Multi-objective clustering problem implementation for jMetal framework.
    Optimizes both WCSS (Within-Cluster Sum of Squares) and Connectivity measures.
    """

    def __init__(self, X: np.ndarray, n_clusters: int):
        """
        Initialize the clustering problem with dataset and desired number of clusters.

        Args:
            X: Input data matrix of shape (n_samples, n_features)
            n_clusters: Number of clusters to form
        """
        super().__init__()

        # Store input data
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.n_clusters = n_clusters

        # Define problem dimensions
        self._number_of_variables = self.n_clusters * self.n_features
        self._number_of_objectives = 2
        self._number_of_constraints = 0

        # Determine bounds for each variable (all centroid coordinates)
        self._lower_bound = []
        self._upper_bound = []
        for i in range(self._number_of_variables):
            feature_idx = i % self.n_features
            self._lower_bound.append(float(np.min(X[:, feature_idx])))
            self._upper_bound.append(float(np.max(X[:, feature_idx])))

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return self._number_of_objectives

    def number_of_constraints(self) -> int:
        return self._number_of_constraints

    def name(self) -> str:
        return "ClusteringProblemMultiObjective"

    def create_solution(self) -> FloatSolution:
        """
        Create a random initial solution (random centroids).

        Returns:
            A new random FloatSolution.
        """
        new_solution = FloatSolution(
            self._lower_bound,
            self._upper_bound,
            self._number_of_objectives,
            self._number_of_constraints
        )

        # Initialize with random values within bounds
        new_solution.variables = [
            np.random.uniform(self._lower_bound[i], self._upper_bound[i])
            for i in range(self._number_of_variables)
        ]

        return new_solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """
        Evaluate a candidate solution against the two objectives:
            1) WCSS (Within-Cluster Sum of Squares)
            2) Connectivity

        Args:
            solution: Candidate cluster centroids encoded as a FloatSolution.

        Returns:
            The solution with updated objective values.
        """
        # Transform solution variables into cluster centroids
        centers = self._decode_centroids(solution.variables)

        # Assign data points to nearest centroids
        labels = self._assign_points_to_clusters(centers)

        # Calculate both objective functions
        wcss_value = self._calculate_wcss(labels, centers)
        connectivity_value = self._calculate_connectivity(labels)

        # Update solution objectives
        solution.objectives[0] = wcss_value
        solution.objectives[1] = connectivity_value

        return solution

    def _decode_centroids(self, variables: list) -> np.ndarray:
        """
        Convert a flat list of variables to a matrix of centroids of shape (n_clusters, n_features).
        """
        return np.array(variables).reshape(self.n_clusters, self.n_features)

    def _assign_points_to_clusters(self, centers: np.ndarray) -> np.ndarray:
        """
        Assign each data point in X to its nearest centroid.
        """
        # Méthode vectorisée pour éviter une boucle supplémentaire
        distances = pairwise_distances(self.X, centers, metric='euclidean')
        return np.argmin(distances, axis=1)

    def _calculate_wcss(self, labels: np.ndarray, centers: np.ndarray) -> float:
        """
        Calculate Within-Cluster Sum of Squares (WCSS).

        Args:
            labels: Assigned cluster for each data point.
            centers: Cluster centers.

        Returns:
            Sum of distances between each point and son centroïde assigné.
        """
        total_wcss = 0.0
        # Distances entre tous les points et tous les centres
        distances = pairwise_distances(self.X, centers, metric='euclidean')

        # Additionner les distances point-centroïde selon l’étiquette
        for i in range(self.n_clusters):
            cluster_indices = (labels == i)
            total_wcss += np.sum(distances[cluster_indices, i])

        return total_wcss

    def _calculate_connectivity(self, labels: np.ndarray, L: int = 10) -> float:
        """
        Calculate the connectivity measure based on nearest neighbors.

        Args:
            labels: Cluster assignments for each data point.
            L: Number of nearest neighbors to consider.

        Returns:
            Connectivity value (lower is better).
        """
        N = self.n_samples

        # Find L nearest neighbors for each point
        n_neighbors = min(L + 1, N)
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        neigh.fit(self.X)

        # On retire l’indice du point lui-même (voisin 0)
        neighbors = neigh.kneighbors(self.X, return_distance=False)[:, 1:]

        # Calculate connectivity
        connectivity = 0.0
        for i in range(N):
            for j, neighbor_idx in enumerate(neighbors[i]):
                if labels[i] != labels[neighbor_idx]:
                    connectivity += 1.0 / (j + 1)

        return connectivity