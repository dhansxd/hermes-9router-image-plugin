"""Hermes directory-plugin entry point; package implementation stays in src."""
from .src.hermes_9router_image import NineRouterImageProvider, register

__all__ = ["NineRouterImageProvider", "register"]
