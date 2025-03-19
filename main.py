import sys
import csv
from table_extraction import process_image
from table_refind import initialize_model, get_ai_response
import re

def extract_table_from_response(response: str):
    """Extracts tabular data by finding the first and last numeric row."""
    lines = response.strip().split("\n")
    table_lines = []

    # Find the first and last valid table rows
    for line in lines:
        if re.search(r'\d', line):  # Check if line contains a number (assumes tables have numbers)
            table_lines.append(line.strip())

    return table_lines

def save_to_csv(response: str, filename: str = "ai_response.csv"):
    """Extracts tabular data and saves it as a CSV file."""
    try:
        # Extract only meaningful table data
        table_lines = extract_table_from_response(response)

        rows = []
        for line in table_lines:
            line = line.strip()
            split_line = [col.strip() for col in line.split(",")]
            rows.append(split_line)

        # Write cleaned data to CSV
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(rows)

        print(f"✅ Response successfully saved to {filename} without extra quotes or unwanted text.")
    
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


def main():
    """Main function to process the image, extract text, get AI response, and save to CSV."""
    if len(sys.argv) != 3:
        print("Usage: python main.py <image_path> <task_id>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    try:
        task_id = int(sys.argv[2])
    except ValueError:
        print("❌ Error: Task ID must be an integer.")
        sys.exit(1)

    try:
        extracted_text = process_image(image_path, task_id)
    except Exception as e:
        print(f"❌ Error extracting text from image: {e}")
        sys.exit(1)

    llm = initialize_model()

    try:
        refined_response = get_ai_response(llm, extracted_text)
        print("AI Response:\n", refined_response)
        save_to_csv(refined_response)
    except Exception as e:
        print(f"❌ Error processing AI response: {e}")

if __name__ == "__main__":
    main()
