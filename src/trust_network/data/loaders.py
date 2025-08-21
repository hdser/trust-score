"""Data loaders for various formats."""

import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Base class for data loading."""
    
    @staticmethod
    def load(data_dir: str, format: str = "csv") -> Dict[str, Any]:
        """Load data from directory based on format."""
        if format == "csv":
            return CSVDataLoader.load_from_directory(data_dir)
        elif format == "json":
            return JSONDataLoader.load_from_directory(data_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")


class CSVDataLoader:
    """Load data from CSV files."""
    
    @staticmethod
    def load_from_directory(data_dir: str) -> Dict[str, Any]:
        """Load network data from CSV files in a directory."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Loading data from CSV files in {data_dir}")
        
        # Load nodes
        nodes = CSVDataLoader._load_nodes(data_path / "nodes.csv")
        
        # Load edges
        edges = CSVDataLoader._load_edges(data_path / "edges.csv")
        
        # Load balances
        balances = CSVDataLoader._load_balances(data_path / "balances.csv")
        
        # Load rates
        rates = CSVDataLoader._load_rates(data_path / "rates.csv")
        
        logger.info(f"Loaded {len(nodes)} nodes, {len(edges)} edges, "
                   f"{len(balances)} balances, {len(rates)} rates")
        
        return {
            "nodes": nodes,
            "edges": edges,
            "balances": balances,
            "rates": rates
        }
    
    @staticmethod
    def _load_nodes(filepath: Path) -> List[Dict[str, Any]]:
        """Load nodes from CSV file."""
        if not filepath.exists():
            logger.warning(f"Nodes file not found: {filepath}")
            return []
        
        df = pd.read_csv(filepath)
        nodes = []
        
        for _, row in df.iterrows():
            metadata = {}
            if 'metadata' in row and pd.notna(row['metadata']):
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid metadata JSON for node {row.get('label', 'unknown')}")
            
            node = {
                "label": str(row['label']),
                "is_converter": bool(row.get('is_converter', False)),
                "token_symbol": str(row.get('token_symbol', '')) if pd.notna(row.get('token_symbol')) else None,
                "metadata": metadata
            }
            nodes.append(node)
        
        return nodes
    
    @staticmethod
    def _load_edges(filepath: Path) -> List[Dict[str, Any]]:
        """Load edges from CSV file."""
        if not filepath.exists():
            logger.warning(f"Edges file not found: {filepath}")
            return []
        
        df = pd.read_csv(filepath)
        edges = []
        
        for _, row in df.iterrows():
            metadata = {}
            if 'metadata' in row and pd.notna(row['metadata']):
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid metadata JSON for edge {row.get('source', 'unknown')}->{row.get('target', 'unknown')}")
            
            edge = {
                "source": str(row['source']),
                "target": str(row['target']),
                "weight": float(row['weight']),
                "metadata": metadata
            }
            edges.append(edge)
        
        return edges
    
    @staticmethod
    def _load_balances(filepath: Path) -> List[Dict[str, Any]]:
        """Load balances from CSV file."""
        if not filepath.exists():
            logger.warning(f"Balances file not found: {filepath}")
            return []
        
        df = pd.read_csv(filepath)
        balances = []
        
        for _, row in df.iterrows():
            balance = {
                "holder": str(row['holder']),
                "token": str(row['token']),
                "amount": float(row['amount'])
            }
            balances.append(balance)
        
        return balances
    
    @staticmethod
    def _load_rates(filepath: Path) -> List[Dict[str, Any]]:
        """Load conversion rates from CSV file."""
        if not filepath.exists():
            logger.warning(f"Rates file not found: {filepath}")
            return []
        
        df = pd.read_csv(filepath)
        rates = []
        
        for _, row in df.iterrows():
            rate = {
                "converter": str(row['converter']),
                "token": str(row['token']),
                "rate": float(row['rate'])
            }
            rates.append(rate)
        
        return rates


class JSONDataLoader:
    """Load data from JSON files."""
    
    @staticmethod
    def load_from_directory(data_dir: str) -> Dict[str, Any]:
        """Load network data from JSON files in a directory."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Loading data from JSON files in {data_dir}")
        
        # Try to load from individual files first
        data = {}
        
        for file_type in ["nodes", "edges", "balances", "rates"]:
            filepath = data_path / f"{file_type}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data[file_type] = json.load(f)
            else:
                data[file_type] = []
        
        # If no individual files, try loading from single network.json
        if all(len(v) == 0 for v in data.values()):
            network_file = data_path / "network.json"
            if network_file.exists():
                with open(network_file, 'r') as f:
                    data = json.load(f)
        
        logger.info(f"Loaded {len(data.get('nodes', []))} nodes, "
                   f"{len(data.get('edges', []))} edges, "
                   f"{len(data.get('balances', []))} balances, "
                   f"{len(data.get('rates', []))} rates")
        
        return data
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """Load network data from a single JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded network data from {filepath}")
        return data