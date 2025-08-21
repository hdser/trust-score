"""Trust Network: A comprehensive trust scoring framework for decentralized token networks."""

# Import the main class
from .core.network import TrustNetwork

# Import data models
from .core.data_models import (
    NodeInfo,
    TrustEdge,
    TokenBalance,
    ConversionRate,
    TrustPath,
    TrustScoreResult,
    NodeTrustScore,
    NetworkStatistics
)

# Import data utilities
from .data.loaders import DataLoader

# Import analysis tools  
from .analysis.analyzer import TrustNetworkAnalyzer

__version__ = "1.0.0"
__author__ = "Trust Network Contributors"

__all__ = [
    "TrustNetwork",
    "NodeInfo",
    "TrustEdge",
    "TokenBalance", 
    "ConversionRate",
    "TrustPath",
    "TrustScoreResult",
    "NodeTrustScore",
    "NetworkStatistics",
    "DataLoader",
    "TrustNetworkAnalyzer"
]