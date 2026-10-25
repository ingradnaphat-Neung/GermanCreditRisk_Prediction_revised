"""Backward-compatible entrypoint; prefer `python train_models.py`."""
from train_models import train_and_save

if __name__ == '__main__':
    from train_models import DEFAULT_OUTPUT
    train_and_save(DEFAULT_OUTPUT)
    print(f'Saved: {DEFAULT_OUTPUT}')
