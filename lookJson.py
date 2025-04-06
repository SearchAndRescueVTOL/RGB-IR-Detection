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

# Example usage
if __name__ == "__main__":
    file_path = "/home/ubuntu/persistent/vtuav-det/VTUAV-det/val_ir.json"  # Change this to your JSON file path
    json_dict = load_json(file_path)
    
    if json_dict:
        print("✅ JSON successfully parsed into a dictionary!")
        print(json_dict)  # Print the parsed dictionary
