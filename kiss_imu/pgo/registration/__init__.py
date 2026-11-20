"""Scan registration backends. Add one by subclassing Registration and registering it
with @register('name'); select it with lo.registration.name in the PGO config."""

from .base import REGISTRY, Registration, RegistrationResult, build_registration, register  # noqa: F401
from . import small_gicp_backend  # noqa: F401  (registers 'small_gicp')
