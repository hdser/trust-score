"""Configuration management for trust network."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlgorithmConfig:
    """Algorithm-specific configuration."""
    social: str = "eigentrust"
    liquidity: str = "hybrid"


@dataclass
class WeightConfig:
    """Weight configuration for composite scores."""
    alpha: float = 0.6
    beta: float = 0.4
    
    def __post_init__(self):
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1):
            raise ValueError("Weights must be in [0, 1]")
        if abs(self.alpha + self.beta - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.0")


@dataclass
class AppleseedConfig:
    """Appleseed algorithm configuration."""
    energy: float = 0.85
    iterations: int = 70
    
    def __post_init__(self):
        if not (0 < self.energy < 1):
            raise ValueError("Appleseed energy must be in (0, 1)")
        if self.iterations <= 0:
            raise ValueError("Iterations must be positive")


@dataclass
class EigenTrustConfig:
    """EigenTrust algorithm configuration."""
    alpha: float = 0.15
    iterations: int = 100
    
    def __post_init__(self):
        if not (0 < self.alpha < 1):
            raise ValueError("EigenTrust alpha must be in (0, 1)")
        if self.iterations <= 0:
            raise ValueError("Iterations must be positive")


@dataclass
class PageRankConfig:
    """PageRank algorithm configuration."""
    damping: float = 0.85
    iterations: int = 100
    
    def __post_init__(self):
        if not (0 < self.damping < 1):
            raise ValueError("PageRank damping must be in (0, 1)")
        if self.iterations <= 0:
            raise ValueError("Iterations must be positive")


@dataclass
class ParameterConfig:
    """Algorithm parameters configuration."""
    tau: float = 0.5
    max_hops: int = 6
    appleseed: AppleseedConfig = field(default_factory=AppleseedConfig)
    eigentrust: EigenTrustConfig = field(default_factory=EigenTrustConfig)
    pagerank: PageRankConfig = field(default_factory=PageRankConfig)
    
    def __post_init__(self):
        if not (0 < self.tau <= 1):
            raise ValueError("Tau must be in (0, 1]")
        if self.max_hops <= 0:
            raise ValueError("Max hops must be positive")


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    enable_caching: bool = True
    cache_size: int = 10000
    enable_parallel: bool = True
    num_workers: int = 4
    
    def __post_init__(self):
        if self.cache_size <= 0:
            raise ValueError("Cache size must be positive")
        if self.num_workers <= 0:
            raise ValueError("Number of workers must be positive")


@dataclass
class ConvergenceConfig:
    """Convergence criteria configuration."""
    tolerance: float = 1e-9
    
    def __post_init__(self):
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    sybil_resistance: bool = True
    pre_trust_converters: bool = True


@dataclass
class TrustNetworkConfig:
    """Main trust network configuration."""
    algorithms: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    parameters: ParameterConfig = field(default_factory=ParameterConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


@dataclass
class DataConfig:
    """Data-related configuration."""
    input_format: str = "csv"
    data_directory: str = "data/sample"
    
    def __post_init__(self):
        if self.input_format not in ["csv", "json"]:
            raise ValueError("Input format must be 'csv' or 'json'")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    
    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")


@dataclass
class NetworkConfig:
    """Complete network configuration."""
    trust_network: TrustNetworkConfig = field(default_factory=TrustNetworkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "NetworkConfig":
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkConfig":
        """Create configuration from dictionary."""
        # Trust network config
        tn_data = data.get("trust_network", {})
        
        # Algorithms
        alg_data = tn_data.get("algorithms", {})
        algorithms = AlgorithmConfig(**alg_data)
        
        # Weights
        weight_data = tn_data.get("weights", {})
        weights = WeightConfig(**weight_data)
        
        # Parameters
        param_data = tn_data.get("parameters", {})
        appleseed = AppleseedConfig(**param_data.get("appleseed", {}))
        eigentrust = EigenTrustConfig(**param_data.get("eigentrust", {}))
        pagerank = PageRankConfig(**param_data.get("pagerank", {}))
        
        parameters = ParameterConfig(
            tau=param_data.get("tau", 0.5),
            max_hops=param_data.get("max_hops", 6),
            appleseed=appleseed,
            eigentrust=eigentrust,
            pagerank=pagerank
        )
        
        # Performance
        perf_data = tn_data.get("performance", {})
        performance = PerformanceConfig(**perf_data)
        
        # Convergence
        conv_data = tn_data.get("convergence", {})
        convergence = ConvergenceConfig(**conv_data)
        
        # Security
        sec_data = tn_data.get("security", {})
        security = SecurityConfig(**sec_data)
        
        # Combine trust network config
        trust_network = TrustNetworkConfig(
            algorithms=algorithms,
            weights=weights,
            parameters=parameters,
            performance=performance,
            convergence=convergence,
            security=security
        )
        
        # Data config
        data_config = DataConfig(**data.get("data", {}))
        
        # Logging config
        logging_config = LoggingConfig(**data.get("logging", {}))
        
        return cls(
            trust_network=trust_network,
            data=data_config,
            logging=logging_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "trust_network": {
                "algorithms": {
                    "social": self.trust_network.algorithms.social,
                    "liquidity": self.trust_network.algorithms.liquidity
                },
                "weights": {
                    "alpha": self.trust_network.weights.alpha,
                    "beta": self.trust_network.weights.beta
                },
                "parameters": {
                    "tau": self.trust_network.parameters.tau,
                    "max_hops": self.trust_network.parameters.max_hops,
                    "appleseed": {
                        "energy": self.trust_network.parameters.appleseed.energy,
                        "iterations": self.trust_network.parameters.appleseed.iterations
                    },
                    "eigentrust": {
                        "alpha": self.trust_network.parameters.eigentrust.alpha,
                        "iterations": self.trust_network.parameters.eigentrust.iterations
                    },
                    "pagerank": {
                        "damping": self.trust_network.parameters.pagerank.damping,
                        "iterations": self.trust_network.parameters.pagerank.iterations
                    }
                },
                "performance": {
                    "enable_caching": self.trust_network.performance.enable_caching,
                    "cache_size": self.trust_network.performance.cache_size,
                    "enable_parallel": self.trust_network.performance.enable_parallel,
                    "num_workers": self.trust_network.performance.num_workers
                },
                "convergence": {
                    "tolerance": self.trust_network.convergence.tolerance
                },
                "security": {
                    "sybil_resistance": self.trust_network.security.sybil_resistance,
                    "pre_trust_converters": self.trust_network.security.pre_trust_converters
                }
            },
            "data": {
                "input_format": self.data.input_format,
                "data_directory": self.data.data_directory
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file
            }
        }


def load_config(config_path: Optional[str] = None) -> NetworkConfig:
    """Load configuration from file or environment."""
    if config_path is None:
        config_path = os.getenv("TRUST_NETWORK_CONFIG_FILE", "config/default.yaml")
    
    if not os.path.exists(config_path):
        # Use default config if file doesn't exist
        return NetworkConfig()
    
    return NetworkConfig.from_yaml(config_path)


def get_data_directory() -> str:
    """Get data directory from environment or config."""
    return os.getenv("TRUST_NETWORK_DATA_DIR", "data/sample")


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging based on configuration."""
    import logging
    
    level = getattr(logging, config.level.upper())
    
    if config.file:
        logging.basicConfig(
            level=level,
            format=config.format,
            filename=config.file
        )
    else:
        logging.basicConfig(
            level=level,
            format=config.format
        )