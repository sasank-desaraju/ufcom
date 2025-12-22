"""
Build script for marimo notebooks.

This script exports marimo notebooks to HTML/WebAssembly format and generates
an index.html file that lists all the notebooks. It handles both regular notebooks
(from the notebooks/ directory) and apps (from the apps/ directory).

The script can be run from the command line with optional arguments:
    uv run .github/scripts/build.py [--output-dir OUTPUT_DIR]

The exported files will be placed in the specified output directory (default: _site).
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///

import subprocess
import shutil
from typing import List, Union
from pathlib import Path

import jinja2
import fire

from loguru import logger

def _export_html_wasm(notebook_path: Path, output_dir: Path, as_app: bool = False) -> bool:
    """Export a single marimo notebook to HTML/WebAssembly format.

    This function takes a marimo notebook (.py file) and exports it to HTML/WebAssembly format.
    If as_app is True, the notebook is exported in "run" mode with code hidden, suitable for
    applications. Otherwise, it's exported in "edit" mode, suitable for interactive notebooks.

    Args:
        notebook_path (Path): Path to the marimo notebook (.py file) to export
        output_dir (Path): Directory where the exported HTML file will be saved
        as_app (bool, optional): Whether to export as an app (run mode) or notebook (edit mode).
                                Defaults to False.

    Returns:
        bool: True if export succeeded, False otherwise
    """
    # Convert .py extension to .html for the output file
    output_path: Path = notebook_path.with_suffix(".html")

    # Base command for marimo export
    # INFO: Marimo version used for export
    # Below version pins a version
    cmd: List[str] = ["uvx", "marimo==0.17.8", "export", "html-wasm", "--sandbox"]
    # Below version uses latest
    # cmd: List[str] = ["uvx", "marimo", "export", "html-wasm", "--sandbox"]

    # Configure export mode based on whether it's an app or a notebook
    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])  # Apps run in "run" mode with hidden code
    else:
        logger.info(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])  # Notebooks run in "edit" mode

    try:
        # Create full output path and ensure directory exists
        output_file: Path = output_dir / notebook_path.with_suffix(".html")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add notebook path and output file to command
        cmd.extend([str(notebook_path), "-o", str(output_file)])

        # Run marimo export command
        logger.debug(f"Running command: {cmd}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully exported {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        # Handle marimo export errors
        logger.error(f"Error exporting {notebook_path}:")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def _generate_index(output_dir: Path,
                    template_file: Path,
                    notebooks_data: List[dict] | None = None,
                    apps_data: List[dict] | None = None,
                    others_data: List[dict] | None = None
                    ) -> None:
    """Generate an index.html file that lists all the notebooks.

    This function creates an HTML index page that displays links to all the exported
    notebooks. The index page includes the marimo logo and displays each notebook
    with a formatted title and a link to open it.

    Args:
        notebooks_data (List[dict]): List of dictionaries with data for notebooks
        apps_data (List[dict]): List of dictionaries with data for apps
        output_dir (Path): Directory where the index.html file will be saved
        template_file (Path, optional): Path to the template file. If None, uses the default template.

    Returns:
        None
    """
    logger.info("Generating index.html")

    # Create the full path for the index.html file
    index_path: Path = output_dir / "index.html"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Set up Jinja2 environment and load template
        template_dir = template_file.parent
        template_name = template_file.name
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"])
        )
        template = env.get_template(template_name)

        # Render the template with notebook and app data
        rendered_html = template.render(notebooks=notebooks_data, apps=apps_data, others=others_data)

        # Write the rendered HTML to the index.html file
        with open(index_path, "w") as f:
            f.write(rendered_html)
        logger.info(f"Successfully generated index.html at {index_path}")

    except IOError as e:
        # Handle file I/O errors
        logger.error(f"Error generating index.html: {e}")
    except jinja2.exceptions.TemplateError as e:
        # Handle template errors
        logger.error(f"Error rendering template: {e}")


def _export(folder: Path, output_dir: Path, as_app: bool=False) -> List[dict]:
    """Export all marimo notebooks in a folder to HTML/WebAssembly format.

    This function finds all Python files in the specified folder and exports them
    to HTML/WebAssembly format using the export_html_wasm function. It returns a
    list of dictionaries containing the data needed for the template.

    Args:
        folder (Path): Path to the folder containing marimo notebooks
        output_dir (Path): Directory where the exported HTML files will be saved
        as_app (bool, optional): Whether to export as apps (run mode) or notebooks (edit mode).

    Returns:
        List[dict]: List of dictionaries with "display_name" and "html_path" for each notebook
    """
    # Check if the folder exists
    if not folder.exists():
        logger.warning(f"Directory not found: {folder}")
        return []

    # Find all Python files recursively in the folder
    notebooks = list(folder.rglob("*.py"))
    logger.debug(f"Found {len(notebooks)} Python files in {folder}")

    # Exit if no notebooks were found
    if not notebooks:
        logger.warning(f"No notebooks found in {folder}!")
        return []

    # For each successfully exported notebook, add its data to the notebook_data list
    notebook_data = [
        {
            "display_name": (nb.stem.replace("_", " ").title()),
            "html_path": str(nb.with_suffix(".html")),
        }
        for nb in notebooks
        if _export_html_wasm(nb, output_dir, as_app=as_app)
    ]

    logger.info(f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files from {folder}")
    return notebook_data

def _copy_static_assets(output_dir: Path, assets_dir: Path = Path("assets")) -> None:
    """Copy static assets to the output directory.

    This function copies all static assets (images, CSS, JS, etc.) from the assets
    directory to the output directory, preserving the directory structure.

    Args:
        output_dir (Path): Directory where the assets will be copied
        assets_dir (Path, optional): Path to the assets directory. Defaults to Path("assets").

    Returns:
        None
    """
    if not assets_dir.exists():
        logger.debug(f"Assets directory not found: {assets_dir}")
        return

    logger.info(f"Copying static assets from {assets_dir} to {output_dir}")

    try:
        # Create destination directory
        dest_dir = output_dir / "assets"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from assets directory
        for item in assets_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_dir / item.name)
                logger.debug(f"Copied {item.name}")
            elif item.is_dir():
                shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
                logger.debug(f"Copied directory {item.name}")

        logger.info("Successfully copied static assets")
    except Exception as e:
        logger.error(f"Error copying static assets: {e}")

def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/tailwind.html.j2",
) -> None:
    """Main function to export marimo notebooks.

    This function:
    1. Parses command line arguments
    2. Exports all marimo notebooks in the 'notebooks' and 'apps' directories
    3. Generates an index.html file that lists all the notebooks

    Command line arguments:
        --output-dir: Directory where the exported files will be saved (default: _site)
        --template: Path to the template file (default: templates/index.html.j2)

    Returns:
        None
    """
    logger.info("Starting marimo build process")

    # Convert output_dir explicitly to Path (not done by fire)
    output_dir: Path = Path(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert template to Path if provided
    template_file: Path = Path(template)
    logger.info(f"Using template file: {template_file}")

    # Export notebooks from the notebooks/ directory
    notebooks_data = _export(Path("notebooks"), output_dir, as_app=False)

    # Export biostats apps from the apps/ directory
    apps_data = _export(Path("apps"), output_dir, as_app=True)

    # Export other apps from the others/ directory
    others_data = _export(Path("others"), output_dir, as_app=True)

    # Exit if no notebooks or apps were found
    if not notebooks_data and not apps_data:
        logger.warning("No notebooks or apps found!")
        return

    # Copy static assets to output directory
    _copy_static_assets(output_dir)

    # Generate the index.html file that lists all notebooks and apps
    # INFO: we are skipping the notebooks now
    # _generate_index(output_dir=output_dir, notebooks_data=notebooks_data, apps_data=apps_data, others_data=others_data, template_file=template_file)
    _generate_index(output_dir=output_dir, apps_data=apps_data, others_data=others_data, template_file=template_file)

    logger.info(f"Build completed successfully. Output directory: {output_dir}")


if __name__ == '__main__':
    fire.Fire(main)
