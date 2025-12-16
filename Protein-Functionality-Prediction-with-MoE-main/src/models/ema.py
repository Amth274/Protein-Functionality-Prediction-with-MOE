"""
Exponential Moving Average (EMA) for model weights.

Maintains shadow parameters that are exponentially-weighted averages of
the model parameters. Helps stabilize training and improve generalization.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains a shadow copy of model parameters that are updated as
    exponentially-weighted averages:
        shadow = decay * shadow + (1 - decay) * param

    Args:
        model: PyTorch model to track
        decay: Decay rate for EMA (default: 0.999)
        device: Device to store shadow parameters
    """

    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device

        # Shadow parameters (EMA weights)
        self.shadow = {}
        # Backup parameters (original weights during evaluation)
        self.backup = {}

        # Initialize shadow parameters
        self._register()

    def _register(self):
        """Register all trainable parameters for EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        """
        Update shadow parameters with current model parameters.

        Called after optimizer.step() to update the EMA weights.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name] -= (1.0 - self.decay) * (
                        self.shadow[name] - param.data.to(self.device)
                    )

    def apply_shadow(self):
        """
        Replace model parameters with shadow parameters.

        Used before evaluation to test with EMA weights.
        Original parameters are backed up.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Backup current parameter
                self.backup[name] = param.data.clone()
                # Replace with shadow
                param.data.copy_(self.shadow[name])

    def restore(self):
        """
        Restore original model parameters.

        Used after evaluation to continue training with original weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': {k: v.cpu() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = {
            k: v.to(self.device) for k, v in state_dict['shadow'].items()
        }


class EMAWrapper:
    """
    Wrapper that makes it easier to use EMA.

    Provides context manager for evaluation:
        with ema_wrapper:
            # Model uses EMA weights
            evaluate(model)
        # Model back to original weights

    Args:
        model: PyTorch model
        decay: EMA decay rate
        enabled: Whether EMA is enabled
    """

    def __init__(self, model, decay=0.999, enabled=True):
        self.enabled = enabled
        if enabled:
            self.ema = EMA(model, decay=decay)
        else:
            self.ema = None

    def update(self):
        """Update EMA weights after optimizer step."""
        if self.enabled and self.ema is not None:
            self.ema.update()

    def __enter__(self):
        """Enter context: apply shadow parameters."""
        if self.enabled and self.ema is not None:
            self.ema.apply_shadow()
        return self

    def __exit__(self, *args):
        """Exit context: restore original parameters."""
        if self.enabled and self.ema is not None:
            self.ema.restore()

    def state_dict(self):
        """Get state for checkpointing."""
        if self.enabled and self.ema is not None:
            return self.ema.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        if self.enabled and self.ema is not None and state_dict:
            self.ema.load_state_dict(state_dict)
