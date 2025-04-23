"""
Custom fingerprinting functions for Hamilton caching.

This module provides custom hash functions for types that cannot be pickled,
such as h5py objects, to enable proper caching in Hamilton pipelines.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
from hamilton.caching import fingerprinting

logger = logging.getLogger(__name__)


@fingerprinting.hash_value.register(h5py.File)
def hash_h5py_file(obj: h5py.File, *args, **kwargs) -> str:
    """Hash an h5py.File object based on its filename and modification time.

    Since h5py.File objects cannot be pickled, we hash based on the file path
    and modification time to uniquely identify the file.

    Args:
        obj: The h5py.File object to hash

    Returns:
        A string hash representing the h5py.File
    """
    try:
        # Get the file path and modification time
        file_path = obj.filename
        mtime = Path(file_path).stat().st_mtime

        # Create a dictionary with the file info
        file_info = {"path": file_path, "mtime": mtime}

        # Hash the file info
        file_info_str = json.dumps(file_info, sort_keys=True)
        return hashlib.md5(file_info_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Error hashing h5py.File: {e}")
        # Fallback to a unique identifier based on object id
        return f"h5py_file_{id(obj)}"


@fingerprinting.hash_value.register(h5py.Dataset)
def hash_h5py_dataset(obj: h5py.Dataset, *args, **kwargs) -> str:
    """Hash an h5py.Dataset object based on its name and shape.

    Args:
        obj: The h5py.Dataset object to hash

    Returns:
        A string hash representing the h5py.Dataset
    """
    try:
        # Get dataset info
        dataset_info = {
            "name": obj.name,
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "file": obj.file.filename if obj.file else None,
        }

        # Hash the dataset info
        dataset_info_str = json.dumps(dataset_info, sort_keys=True)
        return hashlib.md5(dataset_info_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Error hashing h5py.Dataset: {e}")
        # Fallback to a unique identifier based on object id
        return f"h5py_dataset_{id(obj)}"


@fingerprinting.hash_value.register(h5py.Group)
def hash_h5py_group(obj: h5py.Group, *args, **kwargs) -> str:
    """Hash an h5py.Group object based on its name and keys.

    Args:
        obj: The h5py.Group object to hash

    Returns:
        A string hash representing the h5py.Group
    """
    try:
        # Get group info
        group_info = {
            "name": obj.name,
            "keys": list(obj.keys()),
            "file": obj.file.filename if obj.file else None,
        }

        # Hash the group info
        group_info_str = json.dumps(group_info, sort_keys=True)
        return hashlib.md5(group_info_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Error hashing h5py.Group: {e}")
        # Fallback to a unique identifier based on object id
        return f"h5py_group_{id(obj)}"


# Register a hash function for our CitySegData class
# This will be imported and used in cityseg_utils.py
def hash_cityseg_data(obj: Any, *args, **kwargs) -> str:
    """Hash a CitySegData object based on its HDF file path.

    Args:
        obj: The CitySegData object to hash

    Returns:
        A string hash representing the CitySegData
    """
    try:
        # Get the file path and modification time
        file_path = obj.hdf_path
        mtime = Path(file_path).stat().st_mtime

        # Create a dictionary with the file info
        file_info = {"path": str(file_path), "mtime": mtime}

        # Hash the file info
        file_info_str = json.dumps(file_info, sort_keys=True)
        return hashlib.md5(file_info_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Error hashing CitySegData: {e}")
        # Fallback to a unique identifier based on object id
        return f"cityseg_data_{id(obj)}"
