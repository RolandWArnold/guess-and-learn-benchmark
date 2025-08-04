#!/usr/bin/env python

import json
from typing import Any, Dict, List, Union


def get_json_structure(data: Any, indent: int = 0) -> str:
    """
    Recursively parse JSON data and return its skeletal structure.

    Args:
        data: The JSON data to parse
        indent: Current indentation level for formatting

    Returns:
        String representation of the JSON structure
    """
    prefix = "  " * indent

    if isinstance(data, dict):
        if not data:
            return f"{prefix}{{}}"

        result = f"{prefix}{{\n"
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result += f'{prefix}  "{key}": {get_type_info(value)}\n'
                result += get_json_structure(value, indent + 2)
            else:
                result += f'{prefix}  "{key}": {get_type_info(value)}\n'
        result += f"{prefix}}}"
        return result

    elif isinstance(data, list):
        if not data:
            return f"{prefix}[]"

        result = f"{prefix}[\n"
        # Show structure of first element (assuming homogeneous array)
        first_item = data[0]
        if isinstance(first_item, (dict, list)):
            result += f"{prefix}  {get_type_info(first_item)} (array of {len(data)} items)\n"
            result += get_json_structure(first_item, indent + 1)
        else:
            result += f"{prefix}  {get_type_info(first_item)} (array of {len(data)} items)\n"
        result += f"\n{prefix}]"
        return result

    else:
        return f"{prefix}{get_type_info(data)}"


def get_type_info(value: Any) -> str:
    """Get type information for a value."""
    if isinstance(value, dict):
        return f"object ({len(value)} keys)"
    elif isinstance(value, list):
        return f"array ({len(value)} items)"
    elif isinstance(value, str):
        return f"string"
    elif isinstance(value, int):
        return f"number (int)"
    elif isinstance(value, float):
        return f"number (float)"
    elif isinstance(value, bool):
        return f"boolean"
    elif value is None:
        return f"null"
    else:
        return f"unknown type: {type(value).__name__}"


def parse_json_file_structure(filename: str) -> str:
    """
    Parse a JSON file and return its skeletal structure.

    Args:
        filename: Path to the JSON file

    Returns:
        String representation of the JSON structure
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        print(f"JSON Structure for '{filename}':")
        print("=" * 50)
        structure = get_json_structure(data)
        return structure

    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file '{filename}': {e}"
    except Exception as e:
        return f"Error reading file '{filename}': {e}"


def parse_json_string_structure(json_string: str) -> str:
    """
    Parse a JSON string and return its skeletal structure.

    Args:
        json_string: JSON string to parse

    Returns:
        String representation of the JSON structure
    """
    try:
        data = json.loads(json_string)
        print("JSON Structure:")
        print("=" * 30)
        structure = get_json_structure(data)
        return structure

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON string: {e}"


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python json_parser.py <json_file>")
        print("Example: python json_parser.py data.json")
        sys.exit(1)

    filename = sys.argv[1]
    result = parse_json_file_structure(filename)
    print(result)
