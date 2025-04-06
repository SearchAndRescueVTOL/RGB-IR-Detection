import ijson

def analyze_large_json_array(file_path, max_items=100):
    """
    Streams a large JSON file containing an array of objects and analyzes its structure.
    
    Args:
        file_path (str): Path to the JSON file.
        max_items (int): Number of items to sample for structure analysis.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        parser = ijson.items(file, "item")  # Stream JSON array elements
        for i, obj in enumerate(parser):
            if i >= max_items:
                break
            print(f"Object {i + 1}:")
            analyze_json_structure(obj, indent=1)
            print("-" * 50)

# Example usage
if __name__ == "__main__":
    file_path = "large_file.json"  # Change to your file path
    analyze_large_json_array(file_path)
