"""data_prep.py: Centralized data preparation utilities for awkward arrays.

Consolidates repeated patterns for cleaning, flattening, and converting
awkward arrays to zfit-compatible formats. Also includes safe conversion
utilities to reduce try/except boilerplate throughout codebase.
"""

import numpy as np
import awkward as ak
import zfit
from pyutils.pylogger import Logger

# Module-level logger
logger = Logger(print_prefix='[data_prep] ', verbosity=2)



class DataPreparationManager:
    """Manages data cleaning, validation, and conversion to zfit formats.
    
    Consolidates the repeated pattern:
        arr = ak.nan_to_none(arr)
        arr = ak.drop_none(arr)
        np_arr = ak.to_numpy(ak.flatten(arr, axis=None))
        zfit_data = zfit.Data.from_numpy(array=np_arr, obs=obs_space)
    
    into a single, reusable method with consistent error handling.
    """
    
    @staticmethod
    def clean_and_flatten(arr, remove_nans=True):
        """Convert awkward array to clean 1D numpy array.
        
        Args:
            arr: Awkward array (possibly multidimensional with NaNs/Nones)
            remove_nans: Whether to also remove NaN values after flattening
        
        Returns:
            1D numpy array with None/NaN values removed
        
        Raises:
            ValueError: If input is empty or invalid
        """
        if arr is None:
            raise ValueError("Input array is None")
        
        try:
            # Handle awkward arrays
            arr = ak.nan_to_none(arr)
            arr = ak.drop_none(arr)
            np_arr = ak.to_numpy(ak.flatten(arr, axis=None))
            
            if len(np_arr) == 0:
                raise ValueError("Array is empty after cleaning")
            
            # Additional NaN removal if requested
            if remove_nans:
                np_arr = np_arr[~np.isnan(np_arr)]
            
            if len(np_arr) == 0:
                raise ValueError("Array is empty after removing NaNs")
            
            return np_arr
        
        except Exception as e:
            msg = f"Failed to clean and flatten array: {e}"
            if logger:
                logger.log(msg, 'error')
            raise ValueError(msg) from e
    
    @staticmethod
    def to_zfit_data(arr, obs_space, clean=True, name=None):
        """Convert awkward array directly to zfit.Data.
        
        Args:
            arr: Awkward array
            obs_space: zfit.Space object defining the observable range
            clean: Whether to clean/remove NaNs before conversion
            name: Optional name for debugging
        
        Returns:
            zfit.Data object ready for fitting
        
        Raises:
            ValueError: If input invalid or conversion fails
        """
        try:
            np_arr = DataPreparationManager.clean_and_flatten(arr, remove_nans=clean)
            zfit_data = zfit.Data.from_numpy(array=np_arr, obs=obs_space)
            
            if logger and name:
                logger.log(f"Prepared {name}: {len(np_arr)} events", 'info')
            
            return zfit_data
        
        except Exception as e:
            msg = f"Failed to convert array to zfit.Data: {e}"
            if logger:
                logger.log(msg, 'error')
            raise ValueError(msg) from e
    
    @staticmethod
    def get_numpy_array(arr, remove_nans=True):
        """Get clean numpy array without zfit wrapping.
        
        Useful for plotting, binning, etc.
        
        Args:
            arr: Awkward array
            remove_nans: Whether to remove NaN values
        
        Returns:
            1D numpy array
        """
        return DataPreparationManager.clean_and_flatten(arr, remove_nans=remove_nans)
    
    @staticmethod
    def clean_awkward_array(arr, remove_none=True):
        """Clean awkward array in-place without converting to numpy.
        
        Useful when you need to keep awkward structure for counting/broadcasting.
        
        Args:
            arr: Awkward array (possibly with NaN/None values)
            remove_none: Whether to drop None values after nan_to_none conversion
        
        Returns:
            Cleaned awkward array (same structure, values cleaned)
        """
        arr = ak.nan_to_none(arr)
        if remove_none:
            arr = ak.drop_none(arr)
        return arr

    @staticmethod
    def validate_and_report(arr, obs_range=None, name="Array"):
        """Validate array and log statistics.
        
        Args:
            arr: Awkward array
            obs_range: Optional tuple (min, max) for range checking
            name: Name for logging purposes
        
        Returns:
            Cleaned numpy array if valid
        
        Raises:
            ValueError: If validation fails
        """
        try:
            np_arr = DataPreparationManager.clean_and_flatten(arr, remove_nans=True)
            
            stats = {
                'count': len(np_arr),
                'min': np.min(np_arr),
                'max': np.max(np_arr),
                'mean': np.mean(np_arr),
                'std': np.std(np_arr)
            }
            
            if obs_range:
                in_range = np.sum((np_arr >= obs_range[0]) & (np_arr <= obs_range[1]))
                stats['in_range'] = in_range
                stats['out_of_range'] = len(np_arr) - in_range
                
                if stats['out_of_range'] > 0:
                    if logger:
                        logger.log(f"{name}: {stats['out_of_range']} events outside [{obs_range[0]}, {obs_range[1]}]", 'warning')
            
            if logger:
                logger.log(f"{name} statistics: {stats}", 'debug')
            
            return np_arr, stats
        
        except Exception as e:
            msg = f"Validation failed for {name}: {e}"
            if logger:
                logger.log(msg, 'error')
            raise ValueError(msg) from e


# ============================================================================
# Safe Conversion Utilities (reduces try/except boilerplate)
# ============================================================================

def safe_float_conversion(value, default=float('nan')):
    """Safely convert a value to float with default fallback.
    
    Replaces patterns like:
        try:
            result = float(value)
        except:
            result = float('nan')
    
    Args:
        value: Value to convert
        default: Fallback value if conversion fails (default: NaN)
    
    Returns:
        float: Converted value or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_dict_get(obj, key, default=None):
    """Safely get value from dict or object with fallback.
    
    Replaces patterns like:
        try:
            val = obj.get(key) or getattr(obj, key)
        except:
            val = default
    
    Args:
        obj: Dict or object to get value from
        key: Key name
        default: Fallback if key not found
    
    Returns:
        Value or default
    """
    # Try dict access first
    if isinstance(obj, dict):
        return obj.get(key, default)
    
    # Try attribute access
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def safe_numpy_convert(value, dtype=float, default_repr=True):
    """Safely convert value to numpy array with smart fallbacks.
    
    Replaces patterns like:
        try:
            arr = np.asarray(value, dtype=dtype)
        except:
            try:
                arr = np.asarray([list(value)], dtype=object)
            except:
                arr = np.array([repr(value)], dtype=object)
    
    Args:
        value: Value to convert
        dtype: Target dtype (float, int, etc)
        default_repr: If True, use repr() as fallback; else raise
    
    Returns:
        numpy array
    
    Raises:
        ValueError: If conversion fails and default_repr=False
    """
    # Try direct conversion
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        pass
    
    # Try list conversion for sequences
    try:
        return np.asarray([list(value)], dtype=object)
    except (TypeError, ValueError):
        pass
    
    # Use repr as last resort
    if default_repr:
        return np.array([repr(value)], dtype=object)
    
    raise ValueError(f"Cannot convert {type(value)} to numpy array")


def safe_field_extraction(container, *fields):
    """Safely extract nested fields from dict/object with cascading fallbacks.
    
    Useful for hierarchical data extraction with multiple fallback paths.
    
    Example:
        # Instead of:
        if 'trk' in combined.fields:
            fld = combined['trk']
            if 'mom' in ak.fields(fld):
                mom_arr = ak.flatten(fld['mom'], axis=None)
        
        # Use:
        mom_arr = safe_field_extraction(combined, 'trk', 'mom')
    
    Args:
        container: Starting dict/object/awkward array
        *fields: Sequence of field names to extract
    
    Returns:
        Extracted value or None if path not found
    """
    current = container
    
    for field in fields:
        if current is None:
            return None
        
        try:
            # Try dict-like access
            if isinstance(current, dict):
                current = current.get(field)
            # Try awkward array fields
            elif hasattr(current, 'fields'):
                if field in current.fields:
                    current = current[field]
                else:
                    return None
            # Try getitem
            elif hasattr(current, '__getitem__'):
                current = current[field]
            else:
                return None
        except Exception:
            return None
    
    return current

