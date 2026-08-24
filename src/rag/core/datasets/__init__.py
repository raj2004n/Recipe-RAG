"""Dataset subsystem.

Public surface:
    - resolver     : DatasetResolver Protocol + LocalDatasetResolver impl
    - registry_dataset : helpers for registering datasets in the snapshot registry

Import names from the submodule that owns them, e.g.
    from src.rag.core.datasets.resolver import LocalDatasetResolver, ResolvedDataset
    from src.rag.core.datasets.registry_dataset import register_dataset
"""
