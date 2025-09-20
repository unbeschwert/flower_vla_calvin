import logging
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision

import flower
from flower.datasets.utils.episode_utils import load_dataset_statistics

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})


class LeRobotDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for LeRobot datasets.

    This data module handles loading LeRobot format datasets and provides
    train/validation dataloaders compatible with the FLOWER VLA training pipeline.
    """

    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        camera_keys: List[str] = ["observation.images.rgb_static", "observation.images.rgb_left", "observation.images.rgb_right"],
        hf_dataset_name: str = None,  # Hugging Face dataset name
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        self.camera_keys = camera_keys
        self.hf_dataset_name = hf_dataset_name

        # Set up data directories
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(flower.__file__).parent / root_data_path

        self.root_data_path = root_data_path
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms

        # Check if using shared memory datasets
        if 'lang_dataset' in self.datasets_cfg:
            self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset.get('_target_', '')
        else:
            self.use_shm = False

    def prepare_data(self, *args, **kwargs):
        """
        Prepare data (download, extract, etc.).

        For LeRobot datasets, this downloads from Hugging Face Hub if needed.
        """
        if self.hf_dataset_name:
            logger.info(f"Dataset will be downloaded from Hugging Face Hub: {self.hf_dataset_name}")
            logger.info("Download will happen during dataset initialization")
            return

        # Check if local dataset directory exists
        if not self.root_data_path.exists():
            logger.warning(f"Dataset directory {self.root_data_path} does not exist.")
            logger.info("Please ensure your LeRobot dataset is available at the specified path.")
            logger.info("You can download LeRobot datasets from: https://huggingface.co/datasets/lerobot")
            return

        # Check for required metadata files
        for dataset_name in self.datasets_cfg.keys():
            if dataset_name == 'lang_paraphrase-MiniLM-L3-v2':
                continue

            dataset_path = self.root_data_path / dataset_name
            if not dataset_path.exists():
                logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
                continue

            # Check for required LeRobot structure
            required_dirs = ['data', 'meta']
            for req_dir in required_dirs:
                if not (dataset_path / req_dir).exists():
                    logger.warning(f"Missing required directory {req_dir} in {dataset_path}")

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        # Load transforms - for LeRobot we'll create default transforms if none exist
        try:
            # Try to load transforms from existing Calvin/Libero format
            transforms = load_dataset_statistics(
                self.root_data_path / "training",
                self.root_data_path / "validation",
                self.transforms
            )
        except:
            # Create default transforms for LeRobot
            logger.info("Creating default transforms for LeRobot dataset")
            transforms = self._create_default_transforms()

        # Process transforms
        self.train_transforms = {}
        self.val_transforms = {}

        if hasattr(transforms, 'train') and transforms.train:
            for cam in transforms.train:
                cam_transforms = []
                for transform in transforms.train[cam]:
                    if hasattr(transform, '_target_') and transform._target_ == "torchvision.transforms.ColorJitter":
                        instantiated_transform = torchvision.transforms.ColorJitter(
                            brightness=getattr(transform, 'brightness', 0),
                            contrast=tuple(getattr(transform, 'contrast', [1.0, 1.0])),
                            saturation=tuple(getattr(transform, 'saturation', [1.0, 1.0])),
                        )
                    else:
                        instantiated_transform = hydra.utils.instantiate(transform)
                    cam_transforms.append(instantiated_transform)
                self.train_transforms[cam] = torchvision.transforms.Compose(cam_transforms)
        else:
            # Use default transforms for all cameras
            self.train_transforms = self._get_default_camera_transforms()

        if hasattr(transforms, 'val') and transforms.val:
            self.val_transforms = {
                cam: torchvision.transforms.Compose([hydra.utils.instantiate(transform) for transform in transforms.val[cam]])
                for cam in transforms.val
            }
        else:
            self.val_transforms = self._get_default_camera_transforms()

        # Initialize datasets
        self.train_datasets, self.train_sampler = {}, {}
        self.val_datasets, self.val_sampler = {}, {}

        for dataset_name, dataset_config in self.datasets_cfg.items():
            if dataset_name == 'lang_paraphrase-MiniLM-L3-v2':
                continue

            try:
                # Instantiate training dataset
                if self.hf_dataset_name:
                    # Use Hugging Face dataset
                    train_dataset = hydra.utils.instantiate(
                        dataset_config,
                        hf_dataset_name=self.hf_dataset_name,
                        transforms=self.train_transforms,
                        split="train",
                        camera_keys=self.camera_keys,
                    )

                    val_dataset = hydra.utils.instantiate(
                        dataset_config,
                        hf_dataset_name=self.hf_dataset_name,
                        transforms=self.val_transforms,
                        split="validation",
                        camera_keys=self.camera_keys,
                    )
                else:
                    # Use local dataset paths
                    train_dataset_path = self.root_data_path / dataset_name
                    val_dataset_path = self.root_data_path / dataset_name

                    # Check if separate validation directory exists
                    val_dataset_alt_path = self.root_data_path / f"{dataset_name}_validation"
                    if val_dataset_alt_path.exists():
                        val_dataset_path = val_dataset_alt_path

                    if not train_dataset_path.exists():
                        logger.warning(f"Dataset {dataset_name} not found at {train_dataset_path}")
                        continue

                    train_dataset = hydra.utils.instantiate(
                        dataset_config,
                        dataset_path=str(train_dataset_path),
                        transforms=self.train_transforms,
                        split="train",
                        camera_keys=self.camera_keys,
                    )

                    val_dataset = hydra.utils.instantiate(
                        dataset_config,
                        dataset_path=str(val_dataset_path),
                        transforms=self.val_transforms,
                        split="validation",
                        camera_keys=self.camera_keys,
                    )

                key = dataset_config.key
                self.train_datasets[key] = train_dataset
                self.val_datasets[key] = val_dataset
                self.modalities.append(key)

                logger.info(f"Loaded LeRobot dataset {dataset_name} with {len(train_dataset)} train and {len(val_dataset)} val samples")

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue

    def _create_default_transforms(self) -> OmegaConf:
        """Create default transforms for LeRobot datasets."""
        return OmegaConf.create({
            'train': {},
            'val': {}
        })

    def _get_default_camera_transforms(self) -> Dict:
        """Get default transforms for camera observations."""
        default_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Apply same transform to all camera keys
        camera_transforms = {}
        for camera_key in self.camera_keys:
            camera_name = camera_key.split('.')[-1]  # Extract camera name from key
            camera_transforms[camera_name] = default_transform

        return camera_transforms

    def train_dataloader(self):
        """Create training dataloaders."""
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        """Create validation dataloaders."""
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                persistent_workers=True,
                pin_memory=True,
                shuffle=self.shuffle_val,
            )
            for key, dataset in self.val_datasets.items()
        }

    def get_dataset_info(self) -> Dict:
        """Get information about loaded datasets."""
        info = {
            'modalities': self.modalities,
            'num_datasets': len(self.train_datasets),
            'camera_keys': self.camera_keys,
        }

        for key, dataset in self.train_datasets.items():
            if hasattr(dataset, 'info'):
                info[f'{key}_info'] = dataset.info

        return info