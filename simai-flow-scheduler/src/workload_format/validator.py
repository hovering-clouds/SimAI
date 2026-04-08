"""
Workload validator - validates P2P workload files against the schema.
"""

import json
from typing import Optional
from jsonschema import validate, ValidationError, Draft7Validator

from .schema import P2PWorkload, P2P_WORKLOAD_JSON_SCHEMA


class WorkloadValidator:
    """
    Validator for P2P Workload files.

    Performs two levels of validation:
    1. JSON Schema validation - checks structure and required fields
    2. Semantic validation - checks DAG integrity, node ranges, etc.
    """

    def __init__(self, schema: Optional[dict] = None):
        """
        Initialize validator with optional custom schema.

        Args:
            schema: Optional JSON Schema to use. Defaults to P2P_WORKLOAD_JSON_SCHEMA.
        """
        self.schema = schema or P2P_WORKLOAD_JSON_SCHEMA

    def validate_json(self, data: dict) -> list[str]:
        """
        Validate JSON data against schema.

        Args:
            data: Parsed JSON data to validate.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        validator = Draft7Validator(self.schema)
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"Schema validation error at '{path}': {error.message}")

        return errors

    def validate_workload(self, workload: P2PWorkload) -> list[str]:
        """
        Validate a P2PWorkload object.

        Args:
            workload: P2PWorkload object to validate.

        Returns:
            List of validation error messages. Empty if valid.
        """
        return workload.validate()

    def validate_file(self, file_path: str) -> tuple[bool, list[str], Optional[P2PWorkload]]:
        """
        Validate a workload file.

        Args:
            file_path: Path to the JSON file to validate.

        Returns:
            Tuple of (is_valid, errors, workload_or_none)
        """
        errors = []
        workload = None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return False, [f"File not found: {file_path}"], None
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"], None

        # Schema validation
        schema_errors = self.validate_json(data)
        if schema_errors:
            return False, schema_errors, None

        # Deserialize to P2PWorkload
        try:
            workload = self._deserialize(data)
        except Exception as e:
            errors.append(f"Deserialization error: {e}")
            return False, errors, None

        # Semantic validation
        semantic_errors = workload.validate()
        if semantic_errors:
            return False, semantic_errors, workload

        return True, [], workload

    def _deserialize(self, data: dict) -> P2PWorkload:
        """
        Deserialize JSON data to P2PWorkload object.

        Args:
            data: Parsed JSON data.

        Returns:
            P2PWorkload object.
        """
        return P2PWorkload(
            version=data.get("version", "1.0"),
            meta=data.get("meta", {}),
            network=data.get("network"),
            jobs=data.get("jobs", []),
            tasks=data.get("tasks", [])
        )
