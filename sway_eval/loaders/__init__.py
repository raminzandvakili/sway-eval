"""Loaders that turn public datasets into ``Item`` records.

Loaders read files already downloaded to ``data/raw/`` and never modify
them. Readers for parquet need the ``data`` extra: ``uv sync --extra data``.
"""
