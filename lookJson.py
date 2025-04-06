import json

def analyze_json_structure(data, indent=0, visited=None):
    """
    Recursively analyzes the structure of a JSON object.
    
    Args:
        data (dict or list): The JSON object or array to analyze.
        indent (int): The indentation level for pretty-printing.
        visited (set): A set to track visited objects and prevent infinite loops.
    """
    if visited is None:
        visited = set()

    prefix = " " * (indent * 2)  # Indentation for readability

    if isinstance(data, dict):
        print(f"{prefix}Object with {len(data)} keys:")
        for key, value in data.items():
            print(f"{prefix}- {key}: ", end="")
            if isinstance(value, (dict, list)):
                print()  # New line for nested structures
                analyze_json_structure(value, indent + 1, visited)
            else:
                print(type(value).__name__)  # Print type of value
    elif isinstance(data, list):
        print(f"{prefix}Array with {len(data)} elements:")
        if len(data) > 0:
            print(f"{prefix}- First element type: {type(data[0]).__name__}")
            if isinstance(data[0], (dict, list)):
                analyze_json_structure(data[0], indent + 1, visited)
    else:
        print(f"{prefix}{type(data).__name__}")


def load_json_structure(file_path, max_bytes=10_000_000):
    """
    Reads a large JSON file and analyzes its structure without loading it all into memory.
    
    Args:
        file_path (str): Path to the JSON file.
        max_bytes (int): Maximum number of bytes to read at once.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            first_chunk = file.read(max_bytes)  # Read a portion of the file
            data = json.loads(first_chunk)
            analyze_json_structure(data)
        except json.JSONDecodeError:
            print("Error: Could not decode JSON. The file might be too large or malformed.")


# Example usage
if __name__ == "__main__":
    file_path = "large_file.json"  # Change to your file path
    load_json_structure(file_path)
