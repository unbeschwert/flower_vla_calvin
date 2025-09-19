#!/usr/bin/env python3
"""
Test script for LeRobot dataset integration with FLOWER VLA.

This script tests the LeRobot dataset loading pipeline to ensure compatibility
with the FLOWER VLA training framework.

Usage:
    python test_lerobot_dataset.py --dataset_path /path/to/lerobot/dataset
"""

import argparse
import sys
from pathlib import Path

# Add flower to path
sys.path.insert(0, str(Path(__file__).absolute().parent))

import torch
from omegaconf import OmegaConf
from flower.datasets.lerobot_dataset import LeRobotDataset
from flower.datasets.lerobot_data_module import LeRobotDataModule


def test_lerobot_dataset(dataset_path: str = None, hf_dataset_name: str = None):
    """Test LeRobot dataset loading."""
    print("=" * 80)
    print("Testing LeRobot Dataset Loading")
    print("=" * 80)

    # Create test configuration
    obs_space = OmegaConf.create({
        "rgb_obs": ["rgb_static", "rgb_left", "rgb_right"],
        "depth_obs": {},
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"]
    })

    proprio_state = OmegaConf.create({
        "n_state_obs": 18
    })

    # Test dataset initialization
    try:
        if hf_dataset_name:
            print(f"Testing with Hugging Face dataset: {hf_dataset_name}")
            dataset = LeRobotDataset(
                hf_dataset_name=hf_dataset_name,
                obs_space=obs_space,
                proprio_state=proprio_state,
                key="lang",
                lang_folder="",
                num_workers=1,
                transforms={},
                batch_size=4,
                min_window_size=1,
                max_window_size=10,
                pad=True,
                camera_keys=["observation.images.rgb_static", "observation.images.rgb_left", "observation.images.rgb_right"],
                split="train"
            )
        else:
            print(f"Testing with local dataset path: {dataset_path}")
            dataset = LeRobotDataset(
                dataset_path=dataset_path,
                obs_space=obs_space,
                proprio_state=proprio_state,
                key="lang",
                lang_folder="",
                num_workers=1,
                transforms={},
                batch_size=4,
                min_window_size=1,
                max_window_size=10,
                pad=True,
                camera_keys=["observation.images.rgb_static", "observation.images.rgb_left", "observation.images.rgb_right"],
                split="train"
            )

        print(f"✓ Dataset initialized successfully")
        print(f"  - Dataset length: {len(dataset)}")
        print(f"  - Number of episodes: {len(dataset.episodes)}")
        print(f"  - Number of tasks: {len(dataset.tasks)}")

        # Test loading a single sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  - Sample keys: {list(sample.keys())}")

            # Check required keys
            required_keys = ["robot_obs", "actions", "rgb_obs"]
            for key in required_keys:
                if key in sample:
                    if key == "rgb_obs":
                        print(f"  - {key}: {list(sample[key].keys()) if isinstance(sample[key], dict) else 'Not a dict'}")
                    else:
                        print(f"  - {key} shape: {sample[key].shape}")
                else:
                    print(f"  - Missing key: {key}")

        return True

    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_lerobot_datamodule(dataset_path: str = None, hf_dataset_name: str = None):
    """Test LeRobot data module."""
    print("\n" + "=" * 80)
    print("Testing LeRobot DataModule")
    print("=" * 80)

    # Create test configuration
    datasets_config = OmegaConf.create({
        "lang_dataset": {
            "_target_": "flower.datasets.lerobot_dataset.LeRobotDataset",
            "obs_space": {
                "rgb_obs": ["rgb_static", "rgb_left", "rgb_right"],
                "depth_obs": {},
                "state_obs": ["robot_obs"],
                "actions": ["rel_actions"]
            },
            "proprio_state": {
                "n_state_obs": 18
            },
            "key": "lang",
            "batch_size": 2,
            "min_window_size": 1,
            "max_window_size": 10,
            "pad": True,
            "num_workers": 1,
            "transforms": {},
        }
    })

    try:
        if hf_dataset_name:
            datamodule = LeRobotDataModule(
                datasets=datasets_config,
                root_data_dir="./data/lerobot_datasets",
                num_workers=1,
                hf_dataset_name=hf_dataset_name,
                camera_keys=["observation.images.rgb_static", "observation.images.rgb_left", "observation.images.rgb_right"]
            )
        else:
            datamodule = LeRobotDataModule(
                datasets=datasets_config,
                root_data_dir=str(Path(dataset_path).parent),
                num_workers=1,
                camera_keys=["observation.images.rgb_static", "observation.images.rgb_left", "observation.images.rgb_right"]
            )

        print(f"✓ DataModule initialized successfully")

        # Test prepare_data
        datamodule.prepare_data()
        print(f"✓ prepare_data() completed")

        # Test setup
        datamodule.setup()
        print(f"✓ setup() completed")

        # Test dataloaders
        train_dataloaders = datamodule.train_dataloader()
        val_dataloaders = datamodule.val_dataloader()

        print(f"✓ DataLoaders created")
        print(f"  - Train dataloaders: {list(train_dataloaders.keys())}")
        print(f"  - Val dataloaders: {list(val_dataloaders.keys())}")

        # Test loading a batch
        if train_dataloaders:
            for key, dataloader in train_dataloaders.items():
                try:
                    batch = next(iter(dataloader))
                    print(f"✓ Batch loaded from {key} dataloader")
                    print(f"  - Batch keys: {list(batch.keys())}")
                    break
                except Exception as e:
                    print(f"✗ Failed to load batch from {key} dataloader: {e}")

        return True

    except Exception as e:
        print(f"✗ DataModule test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def check_dataset_structure(dataset_path: str):
    """Check if dataset has proper LeRobot structure."""
    print("=" * 80)
    print("Checking Dataset Structure")
    print("=" * 80)

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"✗ Dataset directory does not exist: {dataset_path}")
        return False

    # Check required directories
    required_dirs = ["data", "meta"]
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if dir_path.exists():
            print(f"✓ Found {dir_name}/ directory")
        else:
            print(f"✗ Missing {dir_name}/ directory")
            return False

    # Check required files
    required_files = [
        "meta/info.json",
        "meta/episodes.jsonl",
    ]

    optional_files = [
        "meta/tasks.jsonl",
        "meta/episodes_stats.jsonl",
    ]

    for file_path in required_files:
        full_path = dataset_dir / file_path
        if full_path.exists():
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing required file: {file_path}")
            return False

    for file_path in optional_files:
        full_path = dataset_dir / file_path
        if full_path.exists():
            print(f"✓ Found {file_path} (optional)")
        else:
            print(f"- Missing {file_path} (optional)")

    # Check for data files
    data_dir = dataset_dir / "data"
    parquet_files = list(data_dir.glob("*.parquet"))
    if parquet_files:
        print(f"✓ Found {len(parquet_files)} parquet files in data/")
    else:
        print("✗ No parquet files found in data/")
        return False

    # Check for videos directory
    videos_dir = dataset_dir / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        print(f"✓ Found videos/ directory with {len(video_files)} video files")
    else:
        print("- No videos/ directory found (may not be needed)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test LeRobot dataset integration")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to LeRobot dataset directory (for local datasets)")
    parser.add_argument("--hf_dataset_name", type=str, default="username-007/bimanual-franka-bin-packing",
                        help="Hugging Face dataset name")
    parser.add_argument("--skip_structure_check", action="store_true",
                        help="Skip dataset structure validation")
    parser.add_argument("--skip_dataset_test", action="store_true",
                        help="Skip dataset loading test")
    parser.add_argument("--skip_datamodule_test", action="store_true",
                        help="Skip datamodule test")

    args = parser.parse_args()

    print("LeRobot Dataset Integration Test")
    if args.hf_dataset_name:
        print(f"Hugging Face dataset: {args.hf_dataset_name}")
    else:
        print(f"Local dataset path: {args.dataset_path}")

    success = True

    # Check dataset structure (only for local datasets)
    if not args.skip_structure_check and args.dataset_path:
        if not check_dataset_structure(args.dataset_path):
            success = False

    # Test dataset loading
    if not args.skip_dataset_test and success:
        if not test_lerobot_dataset(args.dataset_path, args.hf_dataset_name):
            success = False

    # Test data module
    if not args.skip_datamodule_test and success:
        if not test_lerobot_datamodule(args.dataset_path, args.hf_dataset_name):
            success = False

    print("\n" + "=" * 80)
    if success:
        print("✓ All tests passed! LeRobot dataset integration is working.")
        print("\nNext steps:")
        if args.hf_dataset_name:
            print("1. Install huggingface_hub: pip install huggingface_hub")
            print("2. Run training with: python flower/training_lerobot.py")
            print(f"3. Dataset '{args.hf_dataset_name}' will be automatically downloaded")
        else:
            print("1. Update conf/config_lerobot.yaml with your dataset path")
            print("2. Adjust camera_keys based on your dataset")
            print("3. Run training with: python flower/training_lerobot.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()