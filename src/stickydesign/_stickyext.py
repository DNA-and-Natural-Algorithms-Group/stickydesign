__all__ = ['fastsub']

try:
    from stickydesign_accel import fastsub
except ImportError:
    raise ImportError("Could not load stickydesign_accel, required for EnergeticsBasicOld.")