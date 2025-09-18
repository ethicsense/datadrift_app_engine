# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.14] - 2025-09-16

### Added
- TensorFlow dependency for complete TensorBoard functionality
- Enhanced path management for PyPI package installation
- Improved project name handling in perturbation results
- Development mode logging for better debugging

### Fixed
- Fixed model and log storage paths to work with both local development and PyPI package installation
- Fixed project name display in perturbation result pages
- Improved error handling for missing project data

### Changed
- Updated path resolution to use `ddoc.flask_webapp` module when available
- Enhanced trainer.py to use absolute paths for consistent log storage

## [1.0.13] - 2025-09-16

### Fixed
- Initial path management improvements

## [1.0.12] - Previous version

### Features
- Initial release with core data drift management functionality
- Flask web application with perturbation analysis
- YOLO model training and evaluation
- FiftyOne integration for dataset management
- CAM visualization capabilities
