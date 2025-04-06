import ijson

def analyze_json_structure(file_path, max_items=5):
    """
    Streams a large JSON file and analyzes its structure without loading it all into memory.

    Args:
        file_path (str): Path to the large JSON file.
        max_items (int): Number of top-level items to analyze (for large files).
    """
    with open(file_path, "r", encoding="utf-8") as file:
        parser = ijson.parse(file)
        structure = {}

        for prefix, event, value in parser:
            keys = prefix.split(".")
            update_structure(structure, keys, event, value)

            # Stop early if we analyzed enough top-level objects
            if keys[0] and structure.get(keys[0], {}).get("__count__", 0) >= max_items:
                break

    print("\n🔍 JSON Structure:")
    print_structure(structure)


def update_structure(structure, keys, event, value):
    """
    Updates the hierarchical structure dictionary based on ijson parsing events.

    Args:
        structure (dict): Dictionary to store JSON structure.
        keys (list): Key path in the JSON.
        event (str): Type of event ('map_key', 'start_map', 'start_array', 'number', etc.).
        value: The value associated with the event (if applicable).
    """
    current = structure
    for key in keys[:-1]:  # Traverse down the key path
        current = current.setdefault(key, {})

    last_key = keys[-1]
    if event == "map_key":
        current[last_key] = current.get(last_key, {})  # Ensure dictionary exists
    elif event == "start_array":
        current[last_key] = {"type": "array", "items": {}}
    elif event == "start_map":
        current[last_key] = {"type": "object"}
    elif event in {"number", "string", "boolean", "null"}:
        current[last_key] = {"type": event, "example": value}

    # Track the count of analyzed items per top-level key
    if keys:
        structure[keys[0]]["__count__"] = structure[keys[0]].get("__count__", 0) + 1


def print_structure(structure, indent=0):
    """
    Recursively prints the JSON structure.

    Args:
        structure (dict): The JSON structure dictionary.
        indent (int): Indentation level for pretty printing.
    """
    prefix = " " * (indent * 2)
    for key, value in structure.items():
        if key == "__count__":
            continue  # Skip internal counters
        if isinstance(value, dict):
            print(f"{prefix}- {key}: {value.get('type', 'object')}")
            print_structure(value, indent + 1)
        else:
            print(f"{prefix}- {key}: {value}")


# Example usage
if __name__ == "__main__":
    file_path = "large_file.json"  # Change to your JSON file
    analyze_json_structure(file_path)