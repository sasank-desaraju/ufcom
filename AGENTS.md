# Agent Instructions for ufcom Repository

## Build/Test/Lint Commands

### Build Commands
- **Full build**: `uv run .github/scripts/build.py` or `uv run build.py`
- **Custom template**: `uv run build.py --template templates/tailwind.html.j2`
- **Custom output dir**: `uv run build.py --output-dir _custom_site`

### Testing Commands
- **Test build locally**: `python -m http.server -d _site` (after running build)
- **No dedicated test runner found** - marimo notebooks are tested via export process

### Linting/Formatting
- **No explicit linter configured** - pre-commit hooks enabled but no config file found
- **Package manager**: `uv` (faster alternative to pip)

## Code Style Guidelines

### Python Style
- **Imports**: Group standard library, then third-party, then local imports
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Type hints**: Use typing module (Union, List, Dict, Optional, etc.)
- **Path handling**: Use `pathlib.Path` instead of strings
- **Docstrings**: Use Google/NumPy style with Args/Returns sections

### Marimo Notebook Conventions
- **Dependencies**: Declare inline with `/// script` comments at file top
- **App structure**: Use `@app.cell` decorators for notebook cells
- **Import pattern**: Import marimo as `mo`, other libs with standard abbreviations
- **Cell organization**: Group related functionality in cells, use descriptive cell comments

### Error Handling
- **Exceptions**: Use try/except blocks with specific exception types
- **Logging**: Use loguru for structured logging with appropriate levels
- **Validation**: Check file/directory existence before operations

### File Structure
- **Notebooks**: Place in `notebooks/` directory (exported in edit mode)
- **Apps**: Place in `apps/` directory (exported in run mode with hidden code)
- **Assets**: Static files in `assets/` directory, data in `*/public/` subdirs
- **Templates**: Jinja2 templates in `templates/` with `.html.j2` extension

### Dependencies
- **Python version**: >= 3.12
- **Key packages**: marimo, altair, pandas, polars, numpy, matplotlib
- **Template engine**: Jinja2 for HTML generation
- **Build tools**: fire (CLI), loguru (logging)