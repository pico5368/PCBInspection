"""Recipe management: load, save, validate product-specific configurations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    """Product-specific inspection configuration."""

    recipe_id: str
    product_name: str
    version: str = "1.0"

    # Alignment
    fiducial_config: dict[str, Any] = field(default_factory=dict)

    # CAD data
    cad_file: str = ""
    cad_format: str = "cpl"  # "cpl" | "odb" | "gerber"

    # Image
    pixels_per_mm: float = 50.0
    image_size: tuple[int, int] = (4000, 3000)
    origin_offset: tuple[float, float] = (0.0, 0.0)

    # Lighting
    lighting_config: dict[str, Any] = field(default_factory=dict)

    # Golden / Model paths
    golden_dir: str = ""
    model_paths: dict[str, str] = field(default_factory=dict)

    # Thresholds (per inspection type)
    thresholds: dict[str, float] = field(default_factory=dict)

    # Component-level overrides
    component_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    recipe = Recipe(
        recipe_id=data.get("recipe_id", path.stem),
        product_name=data.get("product_name", ""),
        version=data.get("version", "1.0"),
        fiducial_config=data.get("fiducial", {}),
        cad_file=data.get("cad_file", ""),
        cad_format=data.get("cad_format", "cpl"),
        pixels_per_mm=data.get("pixels_per_mm", 50.0),
        image_size=tuple(data.get("image_size", [4000, 3000])),
        origin_offset=tuple(data.get("origin_offset", [0.0, 0.0])),
        lighting_config=data.get("lighting", {}),
        golden_dir=data.get("golden_dir", ""),
        model_paths=data.get("model_paths", {}),
        thresholds=data.get("thresholds", {}),
        component_overrides=data.get("component_overrides", {}),
    )

    logger.info("Loaded recipe '%s' v%s", recipe.recipe_id, recipe.version)
    return recipe


def save_recipe(recipe: Recipe, path: str | Path) -> None:
    """Save a recipe to a YAML file."""
    path = Path(path)
    data = {
        "recipe_id": recipe.recipe_id,
        "product_name": recipe.product_name,
        "version": recipe.version,
        "fiducial": recipe.fiducial_config,
        "cad_file": recipe.cad_file,
        "cad_format": recipe.cad_format,
        "pixels_per_mm": recipe.pixels_per_mm,
        "image_size": list(recipe.image_size),
        "origin_offset": list(recipe.origin_offset),
        "lighting": recipe.lighting_config,
        "golden_dir": recipe.golden_dir,
        "model_paths": recipe.model_paths,
        "thresholds": recipe.thresholds,
        "component_overrides": recipe.component_overrides,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logger.info("Saved recipe '%s' to %s", recipe.recipe_id, path)
