import json

def load_json(file_path):
    """
    Loads a JSON file into a Python dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)  # Load JSON as a dictionary
            return data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
def print_dict_tree(d, indent=0):
    """
    Recursively prints the tree structure of a dictionary.

    Args:
        d (dict): The dictionary to print.
        indent (int): Current indentation level.
    """
    prefix = " " * (indent * 2)  # Indentation for tree structure
    if isinstance(d, dict):
        for key, value in d.items():
            print(f"{prefix}- {key}")
            if isinstance(value, (dict, list)):  # Recurse for nested structures
                print_dict_tree(value, indent + 1)
    elif isinstance(d, list):
        for index, item in enumerate(d[:5]):  # Show first 5 items in large lists
            print(f"{prefix}- [{index}]")
            if isinstance(item, (dict, list)):  # Recurse for nested elements
                print_dict_tree(item, indent + 1)
# Example usage
if __name__ == "__main__":
    file_path = "/home/ubuntu/persistent/vtuav-det/VTUAV-det/val_ir.json"  # Change this to your JSON file path
    json_dict = load_json(file_path)
    
    if json_dict:
        print("✅ JSON successfully parsed into a dictionary!")
        print_dict_tree(json_dict)  # Print the parsed dictionary
