from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pretrained_config import PreTrainedPolicyConfig
from flower.models.flower import FLOWERVLA


class FlowerVLAPolicyConfig(PreTrainedPolicyConfig):
    """Configuration for FlowerVLA policy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Extract FLOWERVLA-specific parameters from kwargs
        self.vlm_path = kwargs.get('vlm_path', "microsoft/Florence-2-base")
        self.freeze_florence = kwargs.get('freeze_florence', False)
        self.freeze_vision_tower = kwargs.get('freeze_vision_tower', False)
        self.vlm_prompt_style = kwargs.get('vlm_prompt_style', "default")
        self.token_dropout = kwargs.get('token_dropout', 0.2)
        self.multistep = kwargs.get('multistep', 10)
        self.num_sampling_steps = kwargs.get('num_sampling_steps', 5)
        self.lowdim_obs_dim = kwargs.get('lowdim_obs_dim', 7)
        self.action_dim = kwargs.get('action_dim', 7)
        self.act_window_size = kwargs.get('act_window_size', 10)
        self.use_second_view = kwargs.get('use_second_view', False)
        self.second_view_key = kwargs.get('second_view_key', 'image_wrist')
        self.action_type_adaln = kwargs.get('action_type_adaln', True)
        self.use_causal_attention = kwargs.get('use_causal_attention', True)
        self.use_cross_attn = kwargs.get('use_cross_attn', True)
        self.use_adaln_cond = kwargs.get('use_adaln_cond', False)
        self.use_readout_token = kwargs.get('use_readout_token', False)
        self.use_proprio = kwargs.get('use_proprio', False)
        self.return_act_chunk = kwargs.get('return_act_chunk', False)
        self.sampling_type = kwargs.get('sampling_type', 'ln')
        self.dit_dim = kwargs.get('dit_dim', 512)
        self.n_heads = kwargs.get('n_heads', 16)
        self.n_layers = kwargs.get('n_layers', 12)
        self.attn_pdrop = kwargs.get('attn_pdrop', 0.1)
        self.resid_pdrop = kwargs.get('resid_pdrop', 0.1)
        self.mlp_pdrop = kwargs.get('mlp_pdrop', 0.1)
        self.use_rope = kwargs.get('use_rope', False)
        self.use_nope = kwargs.get('use_nope', False)
        self.query_seq_len = kwargs.get('query_seq_len', 128)
        self.rope_theta = kwargs.get('rope_theta', 32.0)

        # Optimizer and scheduler configs (passed through)
        self.optimizer = kwargs.get('optimizer', {})
        self.lr_scheduler = kwargs.get('lr_scheduler', {})
        self.optimizer_type = kwargs.get('optimizer_type', 'adamw')


class FlowerVLAPolicy(PreTrainedPolicy):
    """LeRobot-compatible wrapper for FLOWERVLA model."""

    name = "flower_vla"
    config_class = FlowerVLAPolicyConfig

    def __init__(self, config: FlowerVLAPolicyConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Create the FLOWERVLA model with configuration
        self.model = FLOWERVLA(
            vlm_path=config.vlm_path,
            freeze_florence=config.freeze_florence,
            freeze_vision_tower=config.freeze_vision_tower,
            vlm_prompt_style=config.vlm_prompt_style,
            token_dropout=config.token_dropout,
            multistep=config.multistep,
            num_sampling_steps=config.num_sampling_steps,
            lowdim_obs_dim=config.lowdim_obs_dim,
            action_dim=config.action_dim,
            act_window_size=config.act_window_size,
            use_second_view=config.use_second_view,
            second_view_key=config.second_view_key,
            action_type_adaln=config.action_type_adaln,
            use_causal_attention=config.use_causal_attention,
            use_cross_attn=config.use_cross_attn,
            use_adaln_cond=config.use_adaln_cond,
            use_readout_token=config.use_readout_token,
            use_proprio=config.use_proprio,
            return_act_chunk=config.return_act_chunk,
            sampling_type=config.sampling_type,
            dit_dim=config.dit_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            mlp_pdrop=config.mlp_pdrop,
            use_rope=config.use_rope,
            use_nope=config.use_nope,
            query_seq_len=config.query_seq_len,
            rope_theta=config.rope_theta,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            optimizer_type=config.optimizer_type,
        )

    def get_optim_params(self):
        """Get optimizer parameters."""
        return self.model._get_param_groups()

    def reset(self):
        """Reset the policy state."""
        self.model.reset()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss during training."""
        converted_batch = self._convert_batch_format(batch)
        obs_features = self.model.encode_observations(converted_batch)
        loss, losses_dict = self.model.rf_loss(obs_features, converted_batch["actions"])
        return {"loss": loss, **losses_dict}

    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict action chunk."""
        converted_batch = self._convert_batch_format(batch)
        obs_features = self.model.encode_observations(converted_batch)

        noise = torch.randn(
            len(obs_features['features']),
            self.model.act_window_size,
            self.model.action_dim,
            device=obs_features['features'].device
        )

        return self.model.sample_actions(noise, obs_features, inference=True)

    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select single action for environment."""
        action_chunk = self.predict_action_chunk(batch)
        return action_chunk[:, 0]

    def _convert_batch_format(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Convert LeRobot batch format to FLOWERVLA format."""
        converted = {
            "rgb_obs": {
                "rgb_static": batch["observation.images.rgb_static"],
                "rgb_left": batch["observation.images.rgb_left"],
                "rgb_right": batch["observation.images.rgb_right"]
            },
            "lang_text": batch.get("instruction", ["default instruction"]),
            "actions": batch.get("action")
        }
        return converted