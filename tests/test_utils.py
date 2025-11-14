"""
Tests for Utility Modules

Tests configuration management, logging, checkpointing, and reproducibility.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_manager import (
    load_config, save_config, merge_configs,
    resolve_config_references, load_experiment_config
)
from src.utils.logger import setup_logger, get_logger
from src.utils.reproducibility import set_seed, get_random_state, get_seed
from src.utils.checkpoint_manager import CheckpointManager


class TestConfigManager:
    """Test configuration management."""

    def test_load_save_config(self, tmp_path):
        """Test loading and saving configs."""
        config = {
            'model': {'type': 'svm', 'C': 1.0},
            'data': {'n_samples': 1000}
        }

        config_path = tmp_path / "test_config.yaml"
        save_config(config, config_path)

        assert config_path.exists()

        loaded = load_config(config_path)
        assert loaded == config

    def test_merge_configs(self):
        """Test merging multiple configs."""
        base = {
            'model': {'type': 'svm', 'C': 1.0},
            'data': {'n_samples': 1000}
        }

        override = {
            'model': {'C': 2.0},
            'data': {'n_samples': 2000, 'seed': 42}
        }

        merged = merge_configs(base, override)

        assert merged['model']['type'] == 'svm'
        assert merged['model']['C'] == 2.0
        assert merged['data']['n_samples'] == 2000
        assert merged['data']['seed'] == 42

    def test_resolve_config_references(self):
        """Test resolving variable references like ${seed}."""
        config = {
            'seed': 42,
            'model': {
                'random_state': '${seed}'
            },
            'training': {
                'random_state': '${seed}'
            }
        }

        resolved = resolve_config_references(config)

        assert resolved['model']['random_state'] == 42
        assert resolved['training']['random_state'] == 42

    def test_load_nonexistent_config(self):
        """Test loading nonexistent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestLogger:
    """Test logging utilities."""

    def test_setup_logger_console(self):
        """Test logger setup with console output."""
        logger = setup_logger(
            name="test_logger",
            level="INFO",
            log_to_file=False
        )

        assert logger.name == "test_logger"
        assert logger.level == 20  # INFO level

    def test_setup_logger_file(self, tmp_path):
        """Test logger setup with file output."""
        logger = setup_logger(
            name="test_logger_file",
            level="DEBUG",
            log_to_file=True,
            log_dir=tmp_path,
            log_file="test.log"
        )

        # Write log message
        logger.info("Test message")

        # Check log file exists
        log_file = tmp_path / "test.log"
        assert log_file.exists()

        # Check message in file
        content = log_file.read_text()
        assert "Test message" in content

    def test_get_logger_cached(self):
        """Test that get_logger returns cached logger."""
        logger1 = setup_logger(name="cached_logger", log_to_file=False)
        logger2 = get_logger("cached_logger")

        assert logger1 is logger2


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_set_seed(self):
        """Test setting random seed."""
        set_seed(42)

        # Generate random numbers
        r1 = np.random.rand(5)

        set_seed(42)

        # Generate again with same seed
        r2 = np.random.rand(5)

        # Should be identical
        np.testing.assert_array_equal(r1, r2)

    def test_get_seed(self):
        """Test getting current seed."""
        set_seed(123)
        assert get_seed() == 123

    def test_get_random_state(self):
        """Test getting RandomState object."""
        set_seed(42)
        rs = get_random_state()

        assert rs is not None
        assert isinstance(rs, np.random.RandomState)


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_checkpoint_creation(self, tmp_path):
        """Test creating checkpoint manager."""
        cm = CheckpointManager(
            save_dir=tmp_path,
            keep_best_only=True,
            metric_name="val_accuracy",
            mode="max"
        )

        assert cm.save_dir == tmp_path
        assert cm.metric_name == "val_accuracy"
        assert cm.mode == "max"

    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        cm = CheckpointManager(
            save_dir=tmp_path,
            keep_best_only=False
        )

        # Mock model
        model = {"weights": [1, 2, 3]}

        # Save checkpoint
        path = cm.save_checkpoint(
            model=model,
            epoch=5,
            metrics={'val_accuracy': 0.85}
        )

        assert path is not None
        assert path.exists()

    def test_save_best_checkpoint(self, tmp_path):
        """Test saving only best checkpoint."""
        cm = CheckpointManager(
            save_dir=tmp_path,
            keep_best_only=True,
            metric_name="val_accuracy",
            mode="max"
        )

        # Mock model
        model = {"weights": [1, 2, 3]}

        # Save first checkpoint (best so far)
        path1 = cm.save_checkpoint(
            model=model,
            epoch=1,
            metrics={'val_accuracy': 0.80}
        )

        assert path1 is not None

        # Save worse checkpoint (should not save)
        path2 = cm.save_checkpoint(
            model=model,
            epoch=2,
            metrics={'val_accuracy': 0.75}
        )

        assert path2 is None

        # Save better checkpoint (should save)
        path3 = cm.save_checkpoint(
            model=model,
            epoch=3,
            metrics={'val_accuracy': 0.90}
        )

        assert path3 is not None
        assert cm.best_metric == 0.90

    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        cm = CheckpointManager(
            save_dir=tmp_path,
            keep_best_only=True
        )

        # Mock model
        model = {"weights": [1, 2, 3]}

        # Save checkpoint
        save_path = cm.save_checkpoint(
            model=model,
            epoch=5,
            metrics={'val_accuracy': 0.85},
            metadata={'lr': 0.001}
        )

        # Load checkpoint
        checkpoint = cm.load_checkpoint(save_path)

        assert checkpoint is not None
        assert checkpoint['model'] == model
        assert checkpoint['epoch'] == 5
        assert checkpoint['metrics']['val_accuracy'] == 0.85
        assert checkpoint['metadata']['lr'] == 0.001

    def test_min_mode(self, tmp_path):
        """Test checkpoint manager in minimize mode."""
        cm = CheckpointManager(
            save_dir=tmp_path,
            keep_best_only=True,
            metric_name="val_loss",
            mode="min"
        )

        model = {"weights": [1, 2, 3]}

        # Save first checkpoint
        cm.save_checkpoint(
            model=model,
            epoch=1,
            metrics={'val_loss': 0.5}
        )

        # Lower loss should be better
        path = cm.save_checkpoint(
            model=model,
            epoch=2,
            metrics={'val_loss': 0.3}
        )

        assert path is not None
        assert cm.best_metric == 0.3

        # Higher loss should not save
        path = cm.save_checkpoint(
            model=model,
            epoch=3,
            metrics={'val_loss': 0.6}
        )

        assert path is None


def test_end_to_end_utilities(tmp_path):
    """Test utilities working together."""
    # Setup logging
    logger = setup_logger(
        name="e2e_test",
        level="INFO",
        log_to_file=True,
        log_dir=tmp_path,
        log_file="e2e.log"
    )

    # Set seed for reproducibility
    set_seed(42)
    logger.info(f"Seed set to: {get_seed()}")

    # Create config
    config = {
        'seed': 42,
        'model': {'type': 'mlp', 'random_state': '${seed}'},
        'data': {'n_samples': 1000}
    }

    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    logger.info(f"Config saved to {config_path}")

    # Load and resolve config
    loaded_config = load_config(config_path)
    resolved_config = resolve_config_references(loaded_config)
    assert resolved_config['model']['random_state'] == 42
    logger.info("Config loaded and resolved")

    # Setup checkpoint manager
    cm = CheckpointManager(save_dir=tmp_path / "checkpoints")

    # Mock training
    for epoch in range(3):
        model = {"weights": np.random.rand(5)}
        metrics = {'val_accuracy': 0.7 + epoch * 0.05}

        cm.save_checkpoint(model, epoch, metrics)
        logger.info(f"Epoch {epoch}: val_acc = {metrics['val_accuracy']:.3f}")

    # Check everything worked
    assert (tmp_path / "e2e.log").exists()
    assert (tmp_path / "config.yaml").exists()
    assert cm.best_checkpoint_path is not None

    logger.info("End-to-end test complete")
