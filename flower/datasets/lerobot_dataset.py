import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torchvision.transforms as transforms

from flower.datasets.base_dataset import BaseDataset
from flower.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


class LeRobotDataset(BaseDataset):
    """
    LeRobot dataset loader compatible with FLOWER VLA.

    This dataset class loads data in LeRobot format and adapts it to work with
    the existing FLOWER VLA training pipeline.

    Args:
        dataset_path: Path to LeRobot dataset directory
        obs_space: DictConfig of observation space
        proprio_state: DictConfig with shape of proprioceptive state
        key: 'vis' or 'lang' (for compatibility with base class)
        lang_folder: Language folder name (not used in LeRobot format)
        num_workers: Number of dataloading workers
        transforms: Dict with pytorch data transforms
        batch_size: Batch size
        min_window_size: Minimum window length of loaded sequences
        max_window_size: Maximum window length of loaded sequences
        pad: If True, pad sequences to max_window_size
        aux_lang_loss_window: Window size for auxiliary language losses
        window_sampling_strategy: 'random' or 'geometric'
        geometric_p_value: P-value for geometric sampling
        camera_keys: List of camera observation keys to load
        split: 'train' or 'validation'
    """

    def __init__(
        self,
        dataset_path: str = None,  # Local path (optional)
        hf_dataset_name: str = None,  # Hugging Face dataset name
        obs_space: DictConfig = None,
        proprio_state: DictConfig = None,
        key: str = "lang",
        lang_folder: str = "",  # Not used for LeRobot
        num_workers: int = 8,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        window_sampling_strategy: str = 'random',
        geometric_p_value: float = 0.1,
        camera_keys: list = ["observation.images.rgb_static", "observation.images.rgb_left", "observation.images.rgb_right"],
        split: str = "train",
        datasets_dir: Optional[Path] = None,  # For compatibility with base class
    ):
        self.hf_dataset_name = hf_dataset_name
        self.dataset_path = None
        self.split = split
        self.camera_keys = camera_keys

        # Handle dataset source - either HF Hub or local path
        if hf_dataset_name:
            # Download from Hugging Face Hub
            self.dataset_path = self._download_from_hf_hub(hf_dataset_name)
        elif dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            raise ValueError("Either hf_dataset_name or dataset_path must be provided")

        # Initialize base class with dummy datasets_dir
        if datasets_dir is None:
            datasets_dir = self.dataset_path / split if split in str(self.dataset_path) else self.dataset_path / f"{split}ing"

        super().__init__(
            datasets_dir=datasets_dir,
            obs_space=obs_space,
            proprio_state=proprio_state,
            key=key,
            lang_folder=lang_folder,
            num_workers=num_workers,
            transforms=transforms,
            batch_size=batch_size,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
            pad=pad,
            aux_lang_loss_window=aux_lang_loss_window,
            window_sampling_strategy=window_sampling_strategy,
            geometric_p_value=geometric_p_value,
        )

        self._load_dataset()

    def _download_from_hf_hub(self, hf_dataset_name: str) -> Path:
        """Download LeRobot dataset from Hugging Face Hub."""
        try:
            from huggingface_hub import snapshot_download
            import os

            # Set cache directory
            cache_dir = Path.home() / ".cache" / "lerobot_datasets"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Check if dataset already exists locally
            local_dataset_path = cache_dir / hf_dataset_name.replace("/", "_")

            if local_dataset_path.exists() and len(list(local_dataset_path.glob("*"))) > 0:
                logger.info(f"Using cached dataset at {local_dataset_path}")
                return local_dataset_path

            logger.info(f"Downloading dataset {hf_dataset_name} from Hugging Face Hub...")

            # Download dataset
            downloaded_path = snapshot_download(
                repo_id=hf_dataset_name,
                repo_type="dataset",
                local_dir=local_dataset_path,
                local_dir_use_symlinks=False,  # Copy files instead of symlinks
            )

            logger.info(f"Dataset downloaded to {downloaded_path}")
            return Path(downloaded_path)

        except ImportError:
            logger.error("huggingface_hub is required to download datasets. Install with: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Failed to download dataset {hf_dataset_name}: {e}")
            raise

    def _load_dataset(self):
        """Load LeRobot dataset metadata and data files."""
        logger.info(f"Loading LeRobot dataset from {self.dataset_path}")

        # Load metadata
        self.info = self._load_json(self.dataset_path / "meta" / "info.json")
        self.episodes = self._load_jsonl(self.dataset_path / "meta" / "episodes.jsonl")

        # Load tasks if available
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            self.tasks = self._load_jsonl(tasks_path)
        else:
            self.tasks = []

        # Load episode stats if available
        stats_path = self.dataset_path / "meta" / "episodes_stats.jsonl"
        if stats_path.exists():
            self.episodes_stats = self._load_jsonl(stats_path)
        else:
            self.episodes_stats = []

        # Load trajectory data from parquet files
        data_dir = self.dataset_path / "data"
        parquet_files = sorted(data_dir.glob("*.parquet"))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Concatenate all parquet files
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)

        self.trajectory_data = pd.concat(dfs, ignore_index=True)

        # Create episode lookup for compatibility with base class
        self._create_episode_lookup()

        # Create language lookup if using language
        if self.with_lang:
            self._create_language_lookup()

        logger.info(f"Loaded {len(self.trajectory_data)} trajectory points from {len(self.episodes)} episodes")

    def _load_json(self, path: Path) -> dict:
        """Load JSON file."""
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _load_jsonl(self, path: Path) -> list:
        """Load JSONL file."""
        if not path.exists():
            return []
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _create_episode_lookup(self):
        """Create episode lookup table for compatibility with base class."""
        self.episode_lookup = self.trajectory_data['episode_index'].values

        # Create frame index lookup for faster access
        self.frame_lookup = self.trajectory_data['frame_index'].values

    def _create_language_lookup(self):
        """Create language lookup for episodes with language annotations."""
        # Create language lookup based on episodes with task descriptions
        self.lang_lookup = []
        for i, episode in enumerate(self.episodes):
            if 'task_index' in episode and episode['task_index'] < len(self.tasks):
                self.lang_lookup.append(episode['task_index'])
            else:
                self.lang_lookup.append(-1)  # No language annotation

        self.lang_lookup = np.array(self.lang_lookup)

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load episode data from LeRobot format.

        Args:
            idx: Starting frame index
            window_size: Number of frames to load

        Returns:
            Episode data dictionary
        """
        # Get episode and frame indices
        episode_idx = self.episode_lookup[idx]
        start_frame = self.frame_lookup[idx]

        # Get trajectory data for this window
        mask = (
            (self.trajectory_data['episode_index'] == episode_idx) &
            (self.trajectory_data['frame_index'] >= start_frame) &
            (self.trajectory_data['frame_index'] < start_frame + window_size)
        )

        window_data = self.trajectory_data[mask].sort_values('frame_index')

        if len(window_data) < window_size:
            # Pad with last frame if needed
            last_row = window_data.iloc[-1:] if len(window_data) > 0 else self.trajectory_data[
                self.trajectory_data['episode_index'] == episode_idx
            ].iloc[:1]

            while len(window_data) < window_size:
                window_data = pd.concat([window_data, last_row], ignore_index=True)

        # Convert to episode format expected by base class
        episode = self._convert_to_episode_format(window_data, episode_idx)

        return episode

    def _convert_to_episode_format(self, window_data: pd.DataFrame, episode_idx: int) -> Dict[str, np.ndarray]:
        """Convert LeRobot data to episode format expected by FLOWER."""
        episode = {}

        # Extract state/proprioceptive observations
        if 'observation.state' in window_data.columns:
            state_data = np.stack([np.array(state) for state in window_data['observation.state']])
            # Extract only TCP poses + grippers (18D) from full 32D state
            # Indices: 0-6 (left TCP), 7-13 (right TCP), 14-17 (grippers)
            tcp_gripper_indices = list(range(14)) + [14, 15, 16, 17]  # TCP poses + grippers
            episode['robot_obs'] = state_data[:, tcp_gripper_indices]
        else:
            # Create dummy state if not available (18D: TCP poses + grippers)
            episode['robot_obs'] = np.zeros((len(window_data), 18))

        # Extract actions
        if 'action' in window_data.columns:
            actions = np.stack([np.array(action) for action in window_data['action']])
            episode['actions'] = actions  # Should be 14D from your data
        else:
            # Create dummy actions if not available (14D: dual arm delta actions)
            episode['actions'] = np.zeros((len(window_data), 14))

        # Extract images for each camera
        episode['rgb_obs'] = {}
        episode['depth_obs'] = {}

        for camera_key in self.camera_keys:
            if camera_key in window_data.columns:
                images = []
                for _, row in window_data.iterrows():
                    img_path = self.dataset_path / "videos" / f"{row[camera_key]}"
                    if img_path.exists():
                        img = self._load_image(img_path)
                        images.append(img)
                    else:
                        # Create dummy image if file not found
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8))

                if images:
                    # Use last part of camera key as camera name
                    camera_name = camera_key.split('.')[-1]
                    episode['rgb_obs'][camera_name] = np.stack(images)

        # Add language if available
        episode_info = self.episodes[episode_idx] if episode_idx < len(self.episodes) else {}
        if 'task_index' in episode_info and episode_info['task_index'] < len(self.tasks):
            task = self.tasks[episode_info['task_index']]
            episode['language'] = task.get('task', task.get('instruction', ''))
        else:
            episode['language'] = ''

        # Add timestamps
        if 'timestamp' in window_data.columns:
            episode['timestamp'] = window_data['timestamp'].values
        else:
            episode['timestamp'] = np.arange(len(window_data))

        return episode

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and process image from file path with Florence-2 preprocessing."""
        if img_path.suffix == '.mp4':
            # Handle video files - extract frame
            # For now, just use the first frame
            import cv2
            cap = cv2.VideoCapture(str(img_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self._preprocess_image(img)
            else:
                return np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Handle image files
            try:
                img = Image.open(img_path)
                return self._preprocess_image(np.array(img))
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                return np.zeros((224, 224, 3), dtype=np.uint8)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for Florence-2: resize to 224x224 and normalize."""
        from PIL import Image
        import torchvision.transforms as T

        # Convert to PIL if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Florence-2 preprocessing: resize and normalize
        transform = T.Compose([
            T.Resize((224, 224)),  # Simple resize to 224x224
            T.ToTensor(),  # Convert to [0, 1] tensor
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transform and convert back to numpy (HWC format)
        tensor = transform(img)
        # Convert from CHW to HWC and denormalize for storage
        img_normalized = tensor.permute(1, 2, 0).numpy()

        # Convert back to uint8 range for compatibility with existing pipeline
        # Note: This will be processed again by the model's transforms
        return ((img_normalized + 1) * 127.5).astype(np.uint8)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.trajectory_data)

    def get_episode_info(self, episode_idx: int) -> dict:
        """Get episode information."""
        if episode_idx < len(self.episodes):
            return self.episodes[episode_idx]
        return {}

    def get_task_info(self, task_idx: int) -> dict:
        """Get task information."""
        if task_idx < len(self.tasks):
            return self.tasks[task_idx]
        return {}