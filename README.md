# TrustScore

Trust scoring framework for Circles Network, implementing multiple trust propagation algorithms and liquidity analysis.


## Features

### **Multiple Trust Algorithms**
- **EigenTrust**: Principal eigenvector-based trust propagation with teleportation
- **Appleseed**: Energy-based trust spreading with configurable decay
- **PageRank**: Web-style authority scoring adapted for trust networks

### **Balance-Aware Liquidity Analysis**
- **Conductance**: Balance-weighted graph-theoretic flow capacity estimation
- **Flow Centrality**: Balance-weighted random walk-based liquidity scoring
- **Hybrid**: Weighted combination of multiple liquidity metrics with funded path checking

### **Payment Path Analysis**
- Trust-based shortest path computation using Dijkstra's algorithm
- Payment capacity estimation implementing theoretical framework
- Bottleneck analysis for route optimization with balance checking
- Funded path verification ensuring viable token transfers

### **Security Analysis**
- Sybil attack resistance measurement with theoretical bounds
- Network resilience metrics and vulnerability assessment
- Community detection and centrality analysis
- Algorithm comparison for security validation

### **Performance Optimized**
- Sparse matrix operations for memory efficiency (O(E) space complexity)
- Intelligent caching with configurable TTL
- Incremental updates for dynamic networks
- Parallel computation support

### **Modular Design**
- Plugin architecture for algorithms with easy extensibility
- Multiple data format support (CSV, JSON)
- Configurable parameters via YAML
- Comprehensive validation and error handling

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- git

### Setup with Virtual Environment

1. **Clone the repository**
```bash
git clone {REPO}
cd trust-score
```

2. **Create and activate Python virtual environment**

**On Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv trust-network-env

# Activate virtual environment
source trust-network-env/bin/activate

# Verify activation (should show the venv path)
which python
```

**On Windows:**
```cmd
# Create virtual environment
python -m venv trust-network-env

# Activate virtual environment
trust-network-env\Scripts\activate

# Verify activation
where python
```

3. **Install dependencies**
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

4. **Verify installation**
```bash
# Test basic import
python -c "from trust_network import TrustNetwork; print('Installation successful!')"

# Run basic example
python examples/basic_usage.py
```

### Alternative: Using conda

```bash
# Create conda environment
conda create -n trust-network python=3.9

# Activate environment
conda activate trust-network

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from trust_network import TrustNetwork
from trust_network.data import DataLoader
from trust_network.config.settings import load_config

# Load configuration and data
config = load_config()
data = DataLoader.load("data/sample", format="csv")

# Create network and compute trust scores
network = TrustNetwork(config)
network.load_data(data)
results = network.compute_trust_scores()

# Get top trusted nodes
top_nodes = results.get_ranking(top_n=10)
for rank, (node, score) in enumerate(top_nodes, 1):
    print(f"{rank}. {node}: {score:.4f}")
```

### Advanced Analysis

```python
from trust_network import TrustNetworkAnalyzer

# Create analyzer for advanced metrics
analyzer = TrustNetworkAnalyzer(network)

# Comprehensive analysis
comprehensive_report = analyzer.generate_comprehensive_report()

# Payment capacity analysis
payment_analysis = analyzer.analyze_payment_bottlenecks()

# Algorithm comparison
algo_comparison = analyzer.compare_algorithms_vs_payment_capacity()

# Network health assessment
health_metrics = analyzer.analyze_network_health_metrics()

print(f"Network health: {health_metrics['health_rating']}")
print(f"Overall score: {health_metrics['overall_score']:.2f}")
```

### Payment Capacity Analysis

```python
# Estimate payment capacity between nodes
capacity = network.estimate_payment_capacity("Alice", "Bob", "Alice")
print(f"Payment capacity: {capacity}")

# Find detailed payment path
path_info = network.find_payment_path_with_capacity("Alice", "Bob", "Alice")
if path_info:
    print(f"Path: {' -> '.join(path_info['path'])}")
    print(f"Capacity: {path_info['capacity']}")
    print(f"Bottlenecks: {len([b for b in path_info['bottlenecks'] if b['is_bottleneck']])}")

# Network liquidity score
liquidity_score = network.compute_network_liquidity_score()
print(f"Network liquidity: {liquidity_score:.3f}")
```

## Configuration

### Configuration File Structure

Configure algorithms and parameters in `config/default.yaml`:

```yaml
trust_network:
  # Algorithm selection
  algorithms:
    social: "eigentrust"     # eigentrust, appleseed, pagerank
    liquidity: "hybrid"      # conductance, pagerank, hybrid
  
  # Composite score weights
  weights:
    alpha: 0.6              # Social component weight (0-1)
    beta: 0.4               # Liquidity component weight (0-1)
  
  # Trust parameters
  parameters:
    tau: 0.5                # Trust acceptance threshold
    max_hops: 6             # Maximum path length
    
    # Algorithm-specific parameters
    eigentrust:
      alpha: 0.15           # Teleportation probability
      iterations: 100       # Maximum iterations
    
    appleseed:
      energy: 0.85          # Energy retention factor
      iterations: 70        # Maximum iterations
    
    pagerank:
      damping: 0.85         # Damping factor
      iterations: 100       # Maximum iterations
  
  # Performance settings
  performance:
    enable_caching: true    # Enable result caching
    cache_size: 10000       # Maximum cache entries
    enable_parallel: true   # Enable parallel computation
    num_workers: 4          # Number of worker threads
```

### Environment Variables

Override configuration using environment variables:

```bash
export TRUST_NETWORK_LOG_LEVEL=DEBUG
export TRUST_NETWORK_DATA_DIR=data/custom
export TRUST_NETWORK_CONFIG_FILE=config/production.yaml
```

## Data Format

The framework supports multiple input formats with comprehensive validation.

### CSV Format (Recommended)

#### `data/sample/nodes.csv`
```csv
label,is_converter,token_symbol,metadata
Alice,false,T_A,"{""type"": ""regular""}"
Bob,false,T_B,"{""type"": ""regular""}"
ConvX,true,,"{""location"": ""US""}"
```

#### `data/sample/edges.csv`
```csv
source,target,weight,metadata
Alice,Bob,0.9,"{""type"": ""friendship""}"
Bob,Charlie,0.8,"{""type"": ""business""}"
ConvX,Alice,1.0,"{""type"": ""service""}"
```

#### `data/sample/balances.csv`
```csv
holder,token,amount
Alice,T_A,1000.0
Alice,T_B,200.0
Bob,T_B,800.0
```

#### `data/sample/rates.csv`
```csv
converter,token,rate
ConvX,T_A,1.00
ConvX,T_B,0.90
ConvY,T_C,0.85
```

### JSON Format

```json
{
  "nodes": [
    {
      "label": "Alice",
      "is_converter": false,
      "token_symbol": "T_A",
      "metadata": {"type": "regular"}
    }
  ],
  "edges": [
    {
      "source": "Alice",
      "target": "Bob", 
      "weight": 0.9,
      "metadata": {"type": "friendship"}
    }
  ],
  "balances": [
    {
      "holder": "Alice",
      "token": "T_A",
      "amount": 1000.0
    }
  ],
  "rates": [
    {
      "converter": "ConvX",
      "token": "T_A",
      "rate": 1.00
    }
  ]
}
```

## Examples

### Running Examples

```bash
# Activate virtual environment
source trust-network-env/bin/activate  # Linux/macOS
# OR
trust-network-env\Scripts\activate     # Windows

# Basic usage example
python examples/basic_usage.py

# Advanced analysis with comprehensive metrics
python examples/advanced_analysis.py

# Performance benchmarking
python examples/performance_test.py

# Theoretical framework validation
python examples/validation_example.py
```

### Example Output

```
============================================================
TRUST NETWORK - COMPREHENSIVE ANALYSIS
============================================================

1. Loading network data...
   - Loaded 11 nodes, 21 edges, 23 balances, 9 rates

2. Computing trust scores...
   - Algorithm: eigentrust+hybrid
   - Computation time: 0.045 seconds

3. Trust Score Rankings:
------------------------------------------------------------
Rank   Node       Composite    Social       Liquidity    
------------------------------------------------------------
1      Alice      0.8234       0.7891       0.8577      
2      ConvX      0.7456       0.8123       0.6789      
3      Bob        0.6789       0.6234       0.7344      

4. Payment Capacity Analysis:
   - Network liquidity score: 0.742
   - Successful payment rate: 68.5%
   - Average path length: 2.3 hops

5. Network Health Assessment:
   - Overall health: Good (0.76)
   - Connectivity: Excellent
   - Balance distribution: Good
   - Converter accessibility: Fair
```

## Architecture

### Algorithm Architecture

The framework uses a plugin-based architecture:

```python
# Social trust algorithms
from trust_network.algorithms.social import get_social_algorithm

social_alg = get_social_algorithm("eigentrust", config)
scores = social_alg.compute(trust_matrix, converters=converters)

# Liquidity algorithms  
from trust_network.algorithms.liquidity import get_liquidity_algorithm

liquidity_alg = get_liquidity_algorithm("hybrid", config)
liquidity = liquidity_alg.compute(trust_matrix, balance_matrix, rate_matrix)
```

### Core Components

- **`TrustNetwork`**: Main orchestration class
- **`TrustNetworkAnalyzer`**: Advanced analysis and metrics
- **`PaymentCapacityCalculator`**: Payment flow analysis
- **Algorithm modules**: Pluggable trust and liquidity algorithms
- **Data handlers**: Loading, validation, and export utilities


## API Documentation

### Core Classes

#### `TrustNetwork`
```python
network = TrustNetwork(config)
network.add_node(label, is_converter=False, **metadata)
network.add_edge(source, target, weight, **metadata)
network.set_balance(holder, token, amount)
network.set_conversion_rate(converter, token, rate)
results = network.compute_trust_scores()
capacity = network.estimate_payment_capacity(source, target, token)
ranking = network.get_ranking(top_n=10, sort_by="composite")
```

#### `TrustNetworkAnalyzer`
```python
analyzer = TrustNetworkAnalyzer(network)
centrality = analyzer.centrality_analyzer.compute_all_centralities()
communities = analyzer.community_analyzer.detect_communities()
bottlenecks = analyzer.analyze_payment_bottlenecks()
health = analyzer.analyze_network_health_metrics()
comparison = analyzer.compare_algorithms_vs_payment_capacity()
```

#### `PaymentCapacityCalculator`
```python
from trust_network.core.payment_capacity import PaymentCapacityCalculator

calculator = PaymentCapacityCalculator(trust_matrix, balance_matrix, tau=0.5)
capacity, path, reason = calculator.compute_payment_capacity(source, target, token)
bottlenecks = calculator.analyze_payment_bottlenecks()
liquidity = calculator.compute_network_liquidity_score()
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Virtual Environment Issues
```bash
# Ensure virtual environment is activated
source trust-network-env/bin/activate  # Linux/macOS
trust-network-env\Scripts\activate     # Windows

# Verify correct environment
which python  # Should show venv path
```

#### 2. Memory Issues for Large Networks
```python
# Adjust configuration for large networks
config.trust_network.performance.enable_caching = False
config.trust_network.algorithms.social = "pagerank"  # Less memory intensive
```

#### 3. Data Format Issues
```python
from trust_network.data import DataValidator

# Validate data before loading
is_valid, errors = DataValidator.validate_network_data(data)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via configuration
config.logging.level = "DEBUG"
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/trust-network.git
cd trust-network

# Create development environment
python -m venv trust-network-dev
source trust-network-dev/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Type checking
mypy src/trust_network/
```

### Adding New Algorithms

```python
# 1. Inherit from appropriate base class
from trust_network.algorithms.base import SocialTrustAlgorithm

class MyAlgorithm(SocialTrustAlgorithm):
    def compute(self, trust_matrix, **kwargs):
        # Implementation here
        return scores

# 2. Register in __init__.py
_ALGORITHMS["my_algorithm"] = MyAlgorithm

# 3. Add tests
def test_my_algorithm():
    # Test implementation
    pass
```

## License

MIT License

