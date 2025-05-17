# BERT Replicate Project

This project uses Python 3.11 with PyTorch and Transformers libraries.

## Setup

1. Make sure you have Python 3.11 installed
2. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   
   uv pip install -r requirements.txt
   ```

## Project Structure

- `pyproject.toml`: Project configuration and metadata
- `requirements.txt`: Project dependencies
- `.python-version`: Python version specification

## Usage

After setting up the environment, you can start using PyTorch and Transformers in your Python code:

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Your code here
``` 