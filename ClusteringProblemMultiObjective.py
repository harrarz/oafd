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
    
    def __init__(self, X, n_clusters):
        """
        Initialize the clustering problem with dataset and desired number of clusters.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            n_clusters: Number of clusters to form
        """
        super().__init__()
        
        # Store input data and problem parameters
        self.X = X
        self.n_clusters = n_clusters
        self.n_features = X.shape[1]
        
        # Define problem dimensions
        self.number_of_variables = n_clusters * self.n_features
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        
        # Determine bounds for each variable (centroids coordinates)
        self.lower_bound = [float(np.min(X[:, i % self.n_features])) for i in range(self.number_of_variables)]
        self.upper_bound = [float(np.max(X[:, i % self.n_features])) for i in range(self.number_of_variables)]

    def number_of_variables(self) -> int:
        return self.number_of_variables
        
    def number_of_objectives(self) -> int:
        return self.number_of_objectives
        
    def number_of_constraints(self) -> int:
        return self.number_of_constraints
        
    def name(self) -> str:
        return "ClusteringProblemMultiObjective"

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """
        Evaluate a candidate solution against the two objectives.
        
        Args:
            solution: Candidate cluster centroids encoded as a FloatSolution
            
        Returns:
            The solution with calculated objective values
        """
        # Transform solution variables into cluster centroids
        centers = self._decode_centroids(solution.variables)
        
        # Assign data points to nearest centroids
        labels = self._assign_points_to_clusters(centers)
        
        # Calculate both objective functions
        wcss_value = self._calculate_wcss(labels, centers)
        conn_value = self._calculate_connectivity(labels)
        
        # Update solution objectives
        solution.objectives[0] = wcss_value
        solution.objectives[1] = conn_value
        
        return solution
    
    def _decode_centroids(self, variables):
        """Convert flat variables array to matrix of centroids."""
        return np.array(variables).reshape(self.n_clusters, self.n_features)
    
    def _assign_points_to_clusters(self, centers):
        """Assign each data point to its nearest centroid."""
        labels = []
        for point in self.X:
            distances = np.linalg.norm(point - centers, axis=1)
            labels.append(np.argmin(distances))
        return np.array(labels)
    
    def _calculate_wcss(self, labels, centers):
        """Calculate Within-Cluster Sum of Squares."""
        total_wcss = 0.0
        for i in range(self.n_clusters):
            cluster_points = self.X[labels == i]
            if len(cluster_points) > 0:
                dists = pairwise_distances(cluster_points, [centers[i]], metric='euclidean')
                total_wcss += np.sum(dists)
        return total_wcss
    
    def _calculate_connectivity(self, labels, L=10):
        """
        Calculate connectivity measure based on nearest neighbors.
        
        Args:
            labels: Cluster assignments for each data point
            L: Number of nearest neighbors to consider
            
        Returns:
            Connectivity value (lower is better)
        """
        N = self.X.shape[0]
        
        # Find L nearest neighbors for each point
        n_neighbors = min(L + 1, N)
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        neigh.fit(self.X)
        neighbors = neigh.kneighbors(self.X, return_distance=False)[:, 1:]  # Skip self as neighbor
        
        # Calculate connectivity
        connectivity = 0.0
        for i in range(N):
            for j, neighbor_idx in enumerate(neighbors[i]):
                if labels[i] != labels[neighbor_idx]:
                    connectivity += 1 / (j + 1)  # Penalty decreases with neighbor distance
                    
        return connectivity

    def create_solution(self) -> FloatSolution:
        """
        Create a random initial solution (random centroids).
        
        Returns:
            A new random solution
        """
        new_solution = FloatSolution(
            self.lower_bound, 
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        
        # Initialize with random values within bounds
        new_solution.variables = [
            np.random.uniform(self.lower_bound[i], self.upper_bound[i])
            for i in range(self.number_of_variables)
        ]
        
        return new_solution