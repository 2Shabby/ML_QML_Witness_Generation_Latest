"""
Checkpoint Manager

Handles saving and loading model checkpoints during training.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Attributes:
        save_dir: Directory for saving checkpoints
        keep_best_only: If True, only keep checkpoint with best metric
        best_metric: Best metric value seen so far
        best_checkpoint_path: Path to best checkpoint
    """

    def __init__(
        self,
        save_dir: Path,
        keep_best_only: bool = True,
        metric_name: str = "val_accuracy",
        mode: str = "max"
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            keep_best_only: Whether to keep only the best checkpoint
            metric_name: Name of metric to track
            mode: "max" or "min" - whether to maximize or minimize metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_only = keep_best_only
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.best_checkpoint_path = None

        logger.info(f"CheckpointManager initialized: {save_dir}")

    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        current_metric = metrics.get(self.metric_name)

        if current_metric is None:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return None

        # Check if this is the best model
        is_best = self._is_better(current_metric)

        if not is_best and self.keep_best_only:
            logger.debug(f"Not saving checkpoint (not best): {current_metric:.4f}")
            return None

        # Create checkpoint
        checkpoint = {
            'model': model,
            'epoch': epoch,
            'metrics': metrics,
            'metadata': metadata or {}
        }

        # Determine save path
        if self.keep_best_only:
            checkpoint_path = self.save_dir / 'best_model.pkl'
        else:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pkl'

        # Save checkpoint
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            logger.info(
                f"Saved checkpoint to {checkpoint_path} "
                f"({self.metric_name}={current_metric:.4f})"
            )

            if is_best:
                self.best_metric = current_metric
                self.best_checkpoint_path = checkpoint_path

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (default: best checkpoint)

        Returns:
            Loaded checkpoint dictionary or None if failed
        """
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _is_better(self, metric: float) -> bool:
        """
        Check if current metric is better than best.

        Args:
            metric: Current metric value

        Returns:
            True if better
        """
        if self.mode == "max":
            return metric > self.best_metric
        else:
            return metric < self.best_metric
