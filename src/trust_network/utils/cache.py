"""Caching utilities for trust computations."""

import time
from typing import Dict, Optional, Any
from collections import OrderedDict
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)


class TrustCache:
    """Cache for trust scores and paths."""
    
    def __init__(self, enabled: bool = True, max_size: int = 10000, ttl: int = 3600):
        self.enabled = enabled
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        
        self._score_cache: OrderedDict = OrderedDict()
        self._path_cache: OrderedDict = OrderedDict()
        self._access_times: Dict[str, float] = {}
    
    def get_scores(self, cache_key: str) -> Optional[Any]:
        """Get cached trust scores."""
        if not self.enabled:
            return None
        
        if cache_key in self._score_cache:
            # Check TTL
            if time.time() - self._access_times.get(cache_key, 0) < self.ttl:
                # Move to end (LRU)
                self._score_cache.move_to_end(cache_key)
                self._access_times[cache_key] = time.time()
                logger.debug(f"Cache hit for scores: {cache_key}")
                return self._score_cache[cache_key]
            else:
                # Expired
                del self._score_cache[cache_key]
                del self._access_times[cache_key]
        
        return None
    
    def set_scores(self, cache_key: str, scores: Any) -> None:
        """Cache trust scores."""
        if not self.enabled:
            return
        
        # Remove oldest entries if cache is full
        while len(self._score_cache) >= self.max_size:
            oldest_key = next(iter(self._score_cache))
            del self._score_cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._score_cache[cache_key] = scores
        self._access_times[cache_key] = time.time()
        logger.debug(f"Cached scores: {cache_key}")
    
    def get_path(self, cache_key: tuple) -> Optional[Any]:
        """Get cached trust path."""
        if not self.enabled:
            return None
        
        key_str = str(cache_key)
        if key_str in self._path_cache:
            path = self._path_cache[key_str]
            # Check TTL
            if time.time() - path.created_at < self.ttl:
                # Move to end (LRU)
                self._path_cache.move_to_end(key_str)
                logger.debug(f"Cache hit for path: {cache_key}")
                return path
            else:
                # Expired
                del self._path_cache[key_str]
        
        return None
    
    def set_path(self, cache_key: tuple, path: Any) -> None:
        """Cache trust path."""
        if not self.enabled:
            return
        
        key_str = str(cache_key)
        
        # Remove oldest entries if cache is full
        while len(self._path_cache) >= self.max_size:
            oldest_key = next(iter(self._path_cache))
            del self._path_cache[oldest_key]
        
        self._path_cache[key_str] = path
        logger.debug(f"Cached path: {cache_key}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self._score_cache.clear()
        self._path_cache.clear()
        self._access_times.clear()
        logger.debug("Cleared all caches")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "score_cache_size": len(self._score_cache),
            "path_cache_size": len(self._path_cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }


class PersistentCache:
    """Persistent cache using file storage."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_scores(self, network_hash: str, scores: Any) -> None:
        """Save scores to persistent storage."""
        filepath = f"{self.cache_dir}/scores_{network_hash}.pkl"
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'scores': scores,
                    'timestamp': time.time()
                }, f)
            logger.debug(f"Saved scores to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save scores to cache: {e}")
    
    def load_scores(self, network_hash: str, max_age: int = 3600) -> Optional[Any]:
        """Load scores from persistent storage."""
        filepath = f"{self.cache_dir}/scores_{network_hash}.pkl"
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check age
            if time.time() - data['timestamp'] < max_age:
                logger.debug(f"Loaded scores from {filepath}")
                return data['scores']
            else:
                logger.debug(f"Cached scores too old, ignoring")
                return None
        except (FileNotFoundError, pickle.UnpicklingError):
            return None
        except Exception as e:
            logger.warning(f"Failed to load scores from cache: {e}")
            return None