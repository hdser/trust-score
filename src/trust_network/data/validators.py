"""Data validation utilities."""

import numpy as np
from typing import Dict, List, Any, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate network data for consistency and correctness."""
    
    @staticmethod
    def validate_network_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate complete network data."""
        errors = []
        
        # Validate nodes
        node_errors = DataValidator.validate_nodes(data.get("nodes", []))
        errors.extend(node_errors)
        
        # Validate edges
        edge_errors = DataValidator.validate_edges(
            data.get("edges", []), 
            data.get("nodes", [])
        )
        errors.extend(edge_errors)
        
        # Validate balances
        balance_errors = DataValidator.validate_balances(
            data.get("balances", []), 
            data.get("nodes", [])
        )
        errors.extend(balance_errors)
        
        # Validate rates
        rate_errors = DataValidator.validate_rates(
            data.get("rates", []), 
            data.get("nodes", [])
        )
        errors.extend(rate_errors)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Data validation failed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("Data validation passed")
        
        return is_valid, errors
    
    @staticmethod
    def validate_nodes(nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate node data."""
        errors = []
        node_labels = set()
        
        for i, node in enumerate(nodes):
            # Check required fields
            if "label" not in node:
                errors.append(f"Node {i}: Missing required field 'label'")
                continue
            
            label = node["label"]
            
            # Check for duplicate labels
            if label in node_labels:
                errors.append(f"Node {i}: Duplicate label '{label}'")
            node_labels.add(label)
            
            # Check label type
            if not isinstance(label, str) or len(label.strip()) == 0:
                errors.append(f"Node {i}: Label must be a non-empty string")
            
            # Check is_converter field
            if "is_converter" in node and not isinstance(node["is_converter"], bool):
                errors.append(f"Node {i}: 'is_converter' must be boolean")
            
            # Check token_symbol
            if "token_symbol" in node and node["token_symbol"] is not None:
                if not isinstance(node["token_symbol"], str):
                    errors.append(f"Node {i}: 'token_symbol' must be string or null")
            
            # Check metadata
            if "metadata" in node and not isinstance(node["metadata"], dict):
                errors.append(f"Node {i}: 'metadata' must be a dictionary")
        
        return errors
    
    @staticmethod
    def validate_edges(edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate edge data."""
        errors = []
        node_labels = {node["label"] for node in nodes if "label" in node}
        
        for i, edge in enumerate(edges):
            # Check required fields
            required_fields = ["source", "target", "weight"]
            for field in required_fields:
                if field not in edge:
                    errors.append(f"Edge {i}: Missing required field '{field}'")
                    continue
            
            if not all(field in edge for field in required_fields):
                continue
            
            source = edge["source"]
            target = edge["target"]
            weight = edge["weight"]
            
            # Check source and target exist
            if source not in node_labels:
                errors.append(f"Edge {i}: Source node '{source}' not found")
            
            if target not in node_labels:
                errors.append(f"Edge {i}: Target node '{target}' not found")
            
            # Check weight validity
            try:
                weight_float = float(weight)
                if not (0 < weight_float <= 1):
                    errors.append(f"Edge {i}: Weight must be in (0, 1], got {weight}")
            except (ValueError, TypeError):
                errors.append(f"Edge {i}: Weight must be a number, got {type(weight)}")
            
            # Check metadata
            if "metadata" in edge and not isinstance(edge["metadata"], dict):
                errors.append(f"Edge {i}: 'metadata' must be a dictionary")
        
        return errors
    
    @staticmethod
    def validate_balances(balances: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate balance data."""
        errors = []
        node_labels = {node["label"] for node in nodes if "label" in node}
        
        for i, balance in enumerate(balances):
            # Check required fields
            required_fields = ["holder", "token", "amount"]
            for field in required_fields:
                if field not in balance:
                    errors.append(f"Balance {i}: Missing required field '{field}'")
                    continue
            
            if not all(field in balance for field in required_fields):
                continue
            
            holder = balance["holder"]
            token = balance["token"]
            amount = balance["amount"]
            
            # Check holder and token exist
            if holder not in node_labels:
                errors.append(f"Balance {i}: Holder '{holder}' not found")
            
            if token not in node_labels:
                errors.append(f"Balance {i}: Token '{token}' not found")
            
            # Check amount validity
            try:
                amount_float = float(amount)
                if amount_float < 0:
                    errors.append(f"Balance {i}: Amount cannot be negative, got {amount}")
            except (ValueError, TypeError):
                errors.append(f"Balance {i}: Amount must be a number, got {type(amount)}")
        
        return errors
    
    @staticmethod
    def validate_rates(rates: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate conversion rate data."""
        errors = []
        node_labels = {node["label"] for node in nodes if "label" in node}
        converters = {node["label"] for node in nodes if node.get("is_converter", False)}
        
        for i, rate in enumerate(rates):
            # Check required fields
            required_fields = ["converter", "token", "rate"]
            for field in required_fields:
                if field not in rate:
                    errors.append(f"Rate {i}: Missing required field '{field}'")
                    continue
            
            if not all(field in rate for field in required_fields):
                continue
            
            converter = rate["converter"]
            token = rate["token"]
            rate_value = rate["rate"]
            
            # Check converter exists and is a converter
            if converter not in node_labels:
                errors.append(f"Rate {i}: Converter '{converter}' not found")
            elif converter not in converters:
                errors.append(f"Rate {i}: Node '{converter}' is not marked as converter")
            
            # Check token exists
            if token not in node_labels:
                errors.append(f"Rate {i}: Token '{token}' not found")
            
            # Check rate validity
            try:
                rate_float = float(rate_value)
                if rate_float < 0:
                    errors.append(f"Rate {i}: Rate cannot be negative, got {rate_value}")
            except (ValueError, TypeError):
                errors.append(f"Rate {i}: Rate must be a number, got {type(rate_value)}")
        
        return errors
    
    @staticmethod
    def check_network_connectivity(edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check network connectivity properties."""
        node_labels = {node["label"] for node in nodes if "label" in node}
        
        # Build adjacency list
        graph = {label: set() for label in node_labels}
        
        for edge in edges:
            if "source" in edge and "target" in edge:
                source = edge["source"]
                target = edge["target"]
                if source in graph and target in graph:
                    graph[source].add(target)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            
            # Visit neighbors (both outgoing and incoming)
            for neighbor in graph[node]:
                dfs(neighbor, component)
            
            # Visit nodes that point to this node
            for other_node, neighbors in graph.items():
                if node in neighbors:
                    dfs(other_node, component)
        
        for node in node_labels:
            if node not in visited:
                component = set()
                dfs(node, component)
                if component:
                    components.append(component)
        
        # Calculate statistics
        largest_component_size = max(len(c) for c in components) if components else 0
        num_isolated_nodes = sum(1 for c in components if len(c) == 1)
        
        return {
            "num_components": len(components),
            "largest_component_size": largest_component_size,
            "num_isolated_nodes": num_isolated_nodes,
            "is_connected": len(components) <= 1,
            "connectivity_ratio": largest_component_size / len(node_labels) if node_labels else 0
        }