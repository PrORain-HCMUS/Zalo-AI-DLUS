# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2025-12-29

### Added

- **Evaluation System**: Created comprehensive evaluation script for model performance assessment
  - `src/evaluate.py` - Full evaluation script with ST-IoU, Precision, Recall, and F1-Score metrics
  - IoU threshold-based matching for accurate performance measurement
  - Spatio-temporal evaluation for video sequences
  - JSON output format for detailed results tracking
  - Support for video-level performance analysis

- **Dataset Configuration**: Added `data.yaml` for YOLO training setup
  - Proper dataset paths for extracted training data
  - Single class configuration for target object detection

### Changed

- **Requirements Update**: Upgraded dependencies to latest compatible versions
  - Updated PyTorch to 2.5.1, torchvision to 0.20.1
  - Updated Ultralytics YOLO to 8.2.0
  - Updated transformers to 4.40.0
  - Added Cython as explicit dependency
  - Commented out problematic packages (lap, cython-bbox) for manual installation

- **Code Structure Improvements**:
  - Fixed import paths in `src/predict.py` and `src/batch_predict.py` for proper module loading
  - Updated file writing operations to use `'w+'` mode for better cross-platform compatibility
  - Added directory creation for output files to prevent path errors

- **Configuration Updates**:
  - Changed default model from `yolov8s.pt` to `yolov8m.pt` in training script
  - Updated config.yaml to use base YOLOv8m model (custom model loading issue noted)
  - Improved dataset path configuration in data.yaml

- **Documentation Enhancements**:
  - Updated README.md with more detailed installation instructions
  - Added proper dataset structure documentation with specific folder names
  - Updated inference commands with actual file paths
  - Fixed evaluation section to reference correct script location
  - Added troubleshooting section for YOLOv8 dataset cache issues

### Fixed

- **Model Loading**: Temporarily using base YOLOv8m model due to custom model compatibility issues
- **Import Errors**: Fixed relative import paths in prediction scripts
- **Path Handling**: Added proper directory creation for output files
- **Installation Issues**: Separated problematic packages for manual installation after core dependencies

### Notes

- Custom trained model from Google Drive currently has compatibility issues
- Using base YOLOv8m model as temporary workaround
- Manual installation required for `lap` and `cython-bbox` packages after base installation

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
