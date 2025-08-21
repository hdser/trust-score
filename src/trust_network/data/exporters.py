"""Data export utilities."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """Export network data and results to various formats."""
    
    @staticmethod
    def export_network(network, filepath: str, format: str = "json") -> None:
        """Export network data to file."""
        if format == "json":
            DataExporter.export_to_json(network.to_dict(), filepath)
        elif format == "csv":
            DataExporter.export_to_csv_files(network.to_dict(), filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    @staticmethod
    def export_results(results, filepath: str, format: str = "json") -> None:
        """Export trust score results to file."""
        if format == "json":
            DataExporter.export_results_to_json(results, filepath)
        elif format == "csv":
            DataExporter.export_results_to_csv(results, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], filepath: str) -> None:
        """Export data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=DataExporter._json_serializer)
        logger.info(f"Exported data to {filepath}")
    
    @staticmethod
    def export_to_csv_files(data: Dict[str, Any], base_path: str) -> None:
        """Export data to multiple CSV files."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Export nodes
        if "nodes" in data:
            nodes_df = pd.DataFrame(data["nodes"])
            nodes_df.to_csv(base_path / "nodes.csv", index=False)
        
        # Export edges
        if "edges" in data:
            edges_df = pd.DataFrame(data["edges"])
            edges_df.to_csv(base_path / "edges.csv", index=False)
        
        # Export balances
        if "balances" in data:
            balances_df = pd.DataFrame(data["balances"])
            balances_df.to_csv(base_path / "balances.csv", index=False)
        
        # Export rates
        if "rates" in data:
            rates_df = pd.DataFrame(data["rates"])
            rates_df.to_csv(base_path / "rates.csv", index=False)
        
        logger.info(f"Exported data to CSV files in {base_path}")
    
    @staticmethod
    def export_results_to_json(results, filepath: str) -> None:
        """Export trust score results to JSON."""
        results_dict = {
            "scores": {node: score.to_dict() for node, score in results.scores.items()},
            "computation_time": results.computation_time,
            "convergence_iterations": results.convergence_iterations,
            "network_stats": {
                "num_nodes": results.network_stats.num_nodes,
                "num_edges": results.network_stats.num_edges,
                "num_converters": results.network_stats.num_converters,
                "total_supply": results.network_stats.total_supply,
                "density": results.network_stats.density,
                "avg_clustering": results.network_stats.avg_clustering,
                "diameter": results.network_stats.diameter
            },
            "algorithm_used": results.algorithm_used,
            "timestamp": results.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=DataExporter._json_serializer)
        logger.info(f"Exported results to {filepath}")
    
    @staticmethod
    def export_results_to_csv(results, filepath: str) -> None:
        """Export trust score results to CSV."""
        # Convert scores to DataFrame
        scores_data = []
        for node, score in results.scores.items():
            scores_data.append(score.to_dict())
        
        df = pd.DataFrame(scores_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported results to {filepath}")
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)