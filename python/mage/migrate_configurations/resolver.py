import yaml
import os
from typing import Any, Dict

import mgp


class Constants:
    CONFIG_FILE_PATH = (
        "/var/lib/memgraph/internal_modules/migrate_config/mongodb_config.yaml"
    )


def _load_configurations() -> Dict[str, Any]:
    """Load configurations from YAML file. Create file if it doesn't exist."""
    config_file = Constants.CONFIG_FILE_PATH

    if not os.path.exists(config_file):
        # Create directory if it doesn't exist
        config_dir = os.path.dirname(config_file)
        try:
            os.makedirs(config_dir, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to create directory {config_dir}: {str(e)}")

        # Create empty YAML file
        try:
            with open(config_file, "w") as file:
                yaml.dump({}, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to create file {config_file}: {str(e)}")

    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        raise Exception(f"Failed to load file {config_file}: {str(e)}")


def _load_configuration_by_name(configuration_name: str) -> Dict[str, Any]:
    """Load a specific configuration by name from YAML file."""
    try:
        configs = _load_configurations()
        return configs.get(configuration_name)
    except Exception:
        return None


def resolve_driver_config(driver_config: mgp.Nullable[mgp.Any]) -> Dict[str, Any]:
    """Resolve driver configuration from either map, string name, or None (auto-load single config)."""
    if driver_config is None:
        # Auto-load single configuration from file
        configs = _load_configurations()
        if len(configs) == 0:
            raise Exception("No configurations found in YAML file. Please use add_configuration to add a configuration.")
        elif len(configs) == 1:
            # Use the single configuration
            return list(configs.values())[0]
        else:
            raise Exception(f"Multiple configurations found ({len(configs)}). Please specify which configuration to use by name or provide a map.")
    elif isinstance(driver_config, str):
        # Load from YAML file by name
        config = _load_configuration_by_name(driver_config)
        if config is None:
            raise Exception(
                f"Configuration '{driver_config}' not found in YAML file. Please use the add_configuration procedure to add a new configuration."
            )
        return config
    elif isinstance(driver_config, mgp.Map):
        # Use the map directly
        return dict(driver_config)
    else:
        raise Exception(
            f"Invalid driver_config type: {type(driver_config)}. Expected a map object with the exact configuration parameters, a string name of the configuration, or None to auto-load single config."
        )
