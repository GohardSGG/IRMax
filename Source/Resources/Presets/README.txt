IRMax installer preset payload directory.

Put factory preset folders/files here before running:
  cargo run -p xtask -- installer --release
or
  cargo run -p xtask -- installer --production

All contents under this directory will be copied by the installer to:
- Windows: %LOCALAPPDATA%\IRMax\Presets
- macOS: ~/Library/Application Support/IRMax/Presets (via postinstall seed)

Notes:
- Existing user files are preserved on macOS (seed only missing files).
- Existing files may be overwritten on Windows if names collide.
