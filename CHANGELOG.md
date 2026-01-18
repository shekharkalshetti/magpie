# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-01-18

### Added

- Parameter validation for monitor decorator
- Comprehensive README with examples and API documentation
- Validation for project_id, custom metadata, and pricing configuration
- Clear error messages for invalid configurations
- GitHub Actions workflow for automated PyPI publishing
- Changelog documentation

### Fixed

- Module import path (magpie_ai → magpie_ai)
- Type hints on decorators now properly show `Callable[[F], F]`
- Package metadata in setup.py
- Complete rebranding from Triton to Magpie (all references updated)
- Environment variables (TRITON*\* → MAGPIE*\*)
- Client class renamed (TritonClient → MagpieClient)

### Changed

- Improved docstrings with Raises section
- Better error handling for conflicting pricing parameters
- All documentation examples now use magpie_ai imports
- Email updated: team@magpie_ai.dev → team@magpie.dev
- URLs updated: magpie_ai.dev → magpie.dev
- Author info updated: Triton Team → Magpie Team

## [0.2.1] - 2025-01-18

### Fixed

- Resolved import path issues (package now correctly exports as magpie_ai)
- Fixed top_level.txt in distribution metadata
- Added proper type hints with py.typed marker

### Added

- Full type support for IDE autocomplete
- Generic type annotations for monitor decorator

## [0.2.0] - 2025-01-18

### Changed

- Renamed package from magpie_ai-sdk to magpie-ai
- Updated all internal module references

### Added

- Proper package naming aligned with PyPI repository name

## [0.1.0] - 2024-12-26

### Added

- Initial release
- Core monitoring decorator (@monitor)
- PII detection and redaction
- Content moderation support
- Automatic token counting and cost calculation
- Context manager for fine-grained monitoring
- Support for OpenAI, Anthropic, and custom LLMs
- Comprehensive logging and metrics collection
