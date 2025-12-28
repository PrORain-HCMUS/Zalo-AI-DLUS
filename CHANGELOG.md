# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-12-28

### Added - Refactor Branch

- **New Project Structure**: Reorganized from `zac2025/` to standard layout
  - `src/` - Source code with modular organization
  - `config/` - Configuration files
  - `checkpoints/` - Model weights directory
  - `docs/` - Documentation
  - `scripts/` - Setup and utility scripts
  - `notebooks/` - Jupyter notebooks
  - `data/` - Dataset directory

- **Source Code Organization**:
  - Split models into separate files: `detector.py`, `feature_extractor.py`, `tracker.py`
  - Updated import paths for new structure
  - Added proper `__init__.py` files with exports
  - Created modular utility functions

- **Documentation**:
  - Comprehensive `README.md` with full reproduction guide
  - `QUICKSTART.md` for quick setup
  - `TRAINING.md` with detailed training instructions
  - `INFERENCE.md` with inference guide
  - Added README files for `data/` and `checkpoints/` directories

- **Configuration**:
  - `requirements.txt` with specific version constraints
  - `config.yaml` with all model and inference parameters
  - `source.txt` with GitHub repository URL

- **Scripts**:
  - `setup.sh` for Linux/Mac setup
  - `setup.bat` for Windows setup

- **Git Configuration**:
  - Updated `.gitignore` for new structure
  - Excluded old `zac2025/` directory
  - Added proper ignores for data, models, and results

### Changed

- Moved code from `zac2025/` to `src/` with updated imports
- Updated README from brief overview to complete reproduction guide
- Reorganized notebooks and documentation into appropriate directories
- Improved code modularity and maintainability

### Fixed

- Import paths now work correctly with new structure
- Configuration paths updated to use `config/` directory
- Model weights path updated to `checkpoints/`

## [0.1.0] - 2025-11-20

### Initial Implementation

- YOLOv8s + DINOv2 + ByteTrack hybrid system
- Basic inference pipeline
- Competition submission format
- Initial documentation in `zac2025/`
