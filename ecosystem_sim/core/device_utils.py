"""GPU/CPU device detection and utility functions.

Detects available compute hardware (NVIDIA CUDA, AMD ROCm, AMD DirectML,
Apple MPS, or CPU) and provides a unified interface for tensor device placement.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache the detected device so we only probe once
_device_type: Optional[str] = None


def detect_device() -> str:
    """Detect available compute device.

    Returns one of: ``"directml"``, ``"rocm"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    The detection order is:
    1. Try importing PyTorch
    2. Check ``torch_directml`` (AMD GPU on Windows via DirectML)
    3. Check ``torch.cuda.is_available()`` — if the GPU name contains ``"AMD"``
       it is classified as ``"rocm"``, otherwise ``"cuda"``
    4. Check Apple Metal Performance Shaders (MPS)
    5. Fall back to ``"cpu"``
    """
    global _device_type
    if _device_type is not None:
        return _device_type

    try:
        import torch

        # 1. Check DirectML first (AMD GPU on Windows)
        try:
            import torch_directml

            if torch_directml.device_count() > 0:
                device_name = torch_directml.device_name(0)
                logger.info("[DEVICE] AMD GPU detected via DirectML: %s", device_name)
                _device_type = "directml"
                return _device_type
        except ImportError:
            pass

        # 2. Check CUDA / ROCm
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "AMD" in device_name or "amd" in device_name.lower():
                logger.info("[DEVICE] AMD GPU detected via ROCm: %s", device_name)
                _device_type = "rocm"
                return _device_type

            logger.info("[DEVICE] NVIDIA GPU detected: %s", device_name)
            _device_type = "cuda"
            return _device_type

        # 3. Apple Metal Performance Shaders
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[DEVICE] Apple Metal (MPS) detected")
            _device_type = "mps"
            return _device_type

    except ImportError:
        logger.info("[DEVICE] PyTorch not installed — falling back to CPU")

    logger.info("[DEVICE] No GPU detected — running on CPU")
    _device_type = "cpu"
    return _device_type


def get_torch_device():
    """Get the optimal PyTorch device for this system.

    Returns:
        A ``torch.device`` instance suitable for tensor placement.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch

    device_type = detect_device()
    if device_type == "directml":
        import torch_directml

        return torch_directml.device()
    elif device_type in ("cuda", "rocm"):
        return torch.device("cuda")
    elif device_type == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """Check whether a GPU (DirectML, CUDA, or ROCm) is available.

    Returns:
        ``True`` if a GPU was detected, ``False`` otherwise.
    """
    return detect_device() in ("directml", "cuda", "rocm")


def device_summary() -> str:
    """Return a human-readable summary of the detected compute device.

    Returns:
        A formatted string describing the device.
    """
    dev = detect_device()
    if dev == "cpu":
        return "CPU"

    try:
        import torch

        if dev == "directml":
            import torch_directml

            name = torch_directml.device_name(0)
            return f"{name} via DirectML"
        elif dev in ("cuda", "rocm"):
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            return f"{name} (Compute Capability {cap[0]}.{cap[1]}) via {dev.upper()}"
        elif dev == "mps":
            return "Apple Metal Performance Shaders (MPS)"
    except ImportError:
        pass

    return dev.upper()
