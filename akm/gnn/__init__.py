"""Graph Neural Network module for link prediction and community detection."""

from akm.gnn.base import GNNManager
from akm.gnn.link_prediction import LinkPredictor

__all__ = ["GNNManager", "LinkPredictor"]
