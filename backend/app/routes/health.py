"""Backward-compatible proxy to app.api.health router"""
from app.api.health import router

__all__ = ["router"]
