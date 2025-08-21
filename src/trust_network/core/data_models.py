"""Data models for trust network components."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import time


@dataclass
class NodeInfo:
    """Information about a network node."""
    label: str
    index: int
    is_converter: bool = False
    token_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class TrustEdge:
    """A directed trust relationship."""
    source: str
    target: str
    weight: float
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 < self.weight <= 1:
            raise ValueError(f"Trust weight must be in (0, 1], got {self.weight}")


@dataclass
class TokenBalance:
    """Token balance for a holder."""
    holder: str
    token: str
    amount: float
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"Balance cannot be negative, got {self.amount}")


@dataclass
class ConversionRate:
    """Conversion rate offered by a converter."""
    converter: str
    token: str
    rate: float  # Fiat value per token
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.rate < 0:
            raise ValueError(f"Conversion rate cannot be negative, got {self.rate}")


@dataclass
class TrustPath:
    """A path of trust between nodes."""
    nodes: List[str]
    trust_value: float
    hop_count: int
    bottleneck_capacity: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    @property
    def is_valid(self) -> bool:
        return self.trust_value > 0 and self.hop_count > 0 and len(self.nodes) > 1


@dataclass
class NodeTrustScore:
    """Trust score components for a single node."""
    node: str
    composite_score: float
    social_score: float
    liquidity_score: float
    
    # Detailed metrics
    eigentrust: float = 0.0
    pagerank: float = 0.0
    appleseed: float = 0.0
    conductance: float = 0.0
    flow_centrality: float = 0.0
    
    # Network position
    in_degree: int = 0
    out_degree: int = 0
    clustering_coefficient: float = 0.0
    converter_distance: int = float('inf')
    
    # Token metrics
    token_supply: float = 0.0
    accessible_supply: float = 0.0
    effective_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node': self.node,
            'composite': round(self.composite_score, 6),
            'social': round(self.social_score, 6),
            'liquidity': round(self.liquidity_score, 6),
            'eigentrust': round(self.eigentrust, 6),
            'pagerank': round(self.pagerank, 6),
            'appleseed': round(self.appleseed, 6),
            'conductance': round(self.conductance, 6),
            'flow_centrality': round(self.flow_centrality, 6),
            'in_degree': self.in_degree,
            'out_degree': self.out_degree,
            'clustering_coefficient': round(self.clustering_coefficient, 6),
            'token_supply': round(self.token_supply, 2),
            'accessible_supply': round(self.accessible_supply, 2),
            'effective_rate': round(self.effective_rate, 4)
        }


@dataclass
class NetworkStatistics:
    """Statistics about the network."""
    num_nodes: int
    num_edges: int
    num_converters: int
    num_tokens: int
    total_supply: float
    avg_clustering: float
    diameter: int
    largest_component_size: int
    density: float


@dataclass
class TrustScoreResult:
    """Complete trust score computation result."""
    scores: Dict[str, NodeTrustScore]
    computation_time: float
    convergence_iterations: int
    network_stats: NetworkStatistics
    algorithm_used: str
    timestamp: float = field(default_factory=time.time)
    
    def get_ranking(self, sort_by: str = "composite", top_n: Optional[int] = None) -> List[tuple]:
        """Get nodes ranked by specified score."""
        if sort_by == "composite":
            scores = [(node, score.composite_score) for node, score in self.scores.items()]
        elif sort_by == "social":
            scores = [(node, score.social_score) for node, score in self.scores.items()]
        elif sort_by == "liquidity":
            scores = [(node, score.liquidity_score) for node, score in self.scores.items()]
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}")
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_n is not None:
            scores = scores[:top_n]
        
        return scores


@dataclass
class GraphMetrics:
    """Graph-level metrics for analysis."""
    degree_centrality: Dict[str, float]
    betweenness_centrality: Dict[str, float]
    closeness_centrality: Dict[str, float]
    eigenvector_centrality: Dict[str, float]
    clustering_coefficients: Dict[str, float]
    
    
@dataclass
class CommunityStructure:
    """Community detection results."""
    communities: Dict[str, int]  # node -> community_id
    modularity: float
    num_communities: int
    community_sizes: Dict[int, int]


@dataclass
class SecurityAnalysis:
    """Security analysis results."""
    sybil_nodes: Set[str]
    total_sybil_score: float
    avg_sybil_score: float
    theoretical_bound: float
    actual_vs_bound_ratio: float
    honest_to_sybil_trust: float
    resistance_effective: bool


@dataclass
class FlowAnalysis:
    """Token flow analysis results."""
    token_velocities: Dict[str, float]
    gini_coefficient: float
    total_supply: float
    avg_velocity: float
    flow_concentration: Dict[str, float]