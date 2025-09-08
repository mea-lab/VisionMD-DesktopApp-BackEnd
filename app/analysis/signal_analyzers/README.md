## Signal Processor README

This guide shows you how to add your own signal processor to the app using the `BaseSignalProcessor` framework.

### Overview

A signal processor:
- Inherits from `BaseSignalProcessor`
- Implements the `analyze` method to process raw signals and return feature dictionaries

## Signal Processor Structure
Inputs:
- raw_signal: List[float]
- normalization_factor: float
- start_time: float
- end_time: float

Returns a dict with at least:
- linePlot: data & time arrays
- peaks, valleys, etc.
- radarTable: summary metrics (mean, std, rate, cv)

### Step 1. Create Your Signal Processor File
Create a new Python file in the `app/analysis/signal_processors` directory, e.g., ` new_signal_processor.py`. Follow the naming convention of other tasks.

### Step 2: Define the Signal Processor Class
```python
from app.analysis.signal_processors.base_signal_processor import BaseSignalProcessor

class MySignalProcessor(BaseSignalProcessor):
    """
    Describe what your processor does (e.g., detects peaks, smooths data).
    """
    @property
    def analyze(self, raw_signal, normalization_factor, start_time, end_time) -> dict:
        # 1) Normalize and prepare your data
        # 2) Compute features (e.g., peaks, velocity)
        # 3) Build and return a dict matching your JSON schema:
        pass
```

### Step 3: Using your new Signal Processor Class
To use your new signal processor class, simply instantiate an object of it within your desired task. 
