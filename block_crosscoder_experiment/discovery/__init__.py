"""SAE discovery, Engels-style topology tests, and the sealed panel.

Zero training. Everything here is a pure function of decoder directions and
harvested codes and are unit-tested against synthetic ground truth.
"""

from block_crosscoder_experiment.discovery.battery import (
    cluster_restricted_reconstruction,
    run_cluster_battery,
    unknown_cluster_scan,
)
from block_crosscoder_experiment.discovery.clustering import (
    angular_similarity,
    cluster_stability,
    coactivation_similarity,
    knn_graph_clusters,
    spectral_clusters,
)
from block_crosscoder_experiment.discovery.indices import (
    epsilon_mixture_index,
    irreducibility_score,
    separability_index,
)
from block_crosscoder_experiment.discovery.nulls import (
    benjamini_hochberg,
    class_permutation_pvalue,
    empirical_pvalue,
    permutation_pvalue,
    random_member_sets,
)
from block_crosscoder_experiment.discovery.rings import (
    angle_harmonic_power,
    circular_decoding,
    cone_normalize,
    ngon_alignment,
    pca_projections,
    plane_scan,
)

__all__ = [
    "angle_harmonic_power",
    "angular_similarity",
    "benjamini_hochberg",
    "circular_decoding",
    "class_permutation_pvalue",
    "cluster_restricted_reconstruction",
    "cluster_stability",
    "coactivation_similarity",
    "cone_normalize",
    "empirical_pvalue",
    "epsilon_mixture_index",
    "irreducibility_score",
    "knn_graph_clusters",
    "ngon_alignment",
    "pca_projections",
    "permutation_pvalue",
    "plane_scan",
    "random_member_sets",
    "run_cluster_battery",
    "separability_index",
    "spectral_clusters",
    "unknown_cluster_scan",
]
