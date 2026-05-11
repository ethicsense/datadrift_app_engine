# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.18] - 2025-10-13

### Fixed
- **Clustering Analysis Robustness**
  - Fixed `TypeError: 'float' object is not subscriptable` in `embedding_analyzer.py`
  - Added type checking for `centroid_similarities` to handle both array and scalar values
  - Improved error handling when similarities data is malformed or empty
  - Added array bounds checking to prevent index out of range errors
  - Enhanced `top_similar_files` calculation with proper type validation

### Changed
- **Cache Filename Structure**
  - Simplified cache filename format by removing redundant "analysis_" prefix
  - Changed from `analysis_attribute_analysis_test_data.cache` to `attribute_analysis_test_data.cache`
  - Baseline files now use `baseline_` prefix: `baseline_attribute_analysis_test_data.cache`
  - Improved cache file naming consistency across the system
  
- **Cluster Statistics Calculation**
  - Made cluster statistics calculation more robust with defensive type checking
  - Gracefully handle edge cases where similarity scores are not available
  - Default to empty lists/zero values when centroid similarities are invalid

## [1.0.17] - 2025-01-27

### Added
- **Netron Package Integration**
  - Added netron package to build dependencies for neural network model visualization
  - Enhanced model visualization capabilities for better model analysis and debugging

## [1.0.16] - 2025-01-27

### Added
- **Perturbation Edit Project Deletion Feature**
  - Added comparison history deletion functionality in perturbation edit page
  - New API endpoint `/api/perturbation/comparison/delete` for deleting individual comparison results
  - Delete button with confirmation dialog for each comparison in edit history
  - Automatic refresh of comparison list after deletion
  - Enhanced UI with delete buttons alongside view buttons

### Changed
- **Terminology Standardization**
  - Renamed all "re-edit" references to "edit" throughout the project for consistency
  - Updated API endpoints: `/api/perturbation/re-edit` → `/api/perturbation/edit`
  - Updated API endpoints: `/api/perturbation/get-re-edit-data` → `/api/perturbation/get-edit-data`
  - Updated API endpoints: `/api/perturbation/check-re-edit-data` → `/api/perturbation/check-edit-data`
  - Updated route: `/perturbation/re-edit-page` → `/perturbation/edit`
  - Updated function names and variable names for consistency
  - Updated UI text and comments to use "edit" terminology

### Fixed
- Removed duplicate route definitions in Flask application
- Fixed template references after terminology changes
- Improved error handling in comparison deletion functionality

## [1.0.15] - 2025-01-20

### Added
- **TensorBoard Logging Enhancements**
  - Comprehensive TensorBoard integration for model training monitoring
  - Real-time loss and accuracy tracking during training sessions
  - Model graph visualization and histogram logging
  - Enhanced logging configuration for better debugging and monitoring
  - Training metrics visualization in TensorBoard dashboard

### Fixed
- **TensorBoard Integration Issues**
  - Fixed TensorBoard log directory path resolution
  - Resolved logging conflicts between different training sessions
  - Fixed model checkpoint saving with proper TensorBoard integration
  - Improved log file management and cleanup
  - Fixed TensorBoard server startup and port configuration

### Changed
- **Training Infrastructure Improvements**
  - Enhanced trainer.py with better TensorBoard logging capabilities
  - Improved model training loop with comprehensive metrics tracking
  - Updated logging configuration for better performance monitoring
  - Enhanced error handling during training sessions

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
