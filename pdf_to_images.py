"""
pdf_to_images.py
----------------
Convert PDF pages to high-resolution JPG files for YOLO training.

This script:
1. Reads PDF files from "Production drawings (pdfs)" folder
2. Converts each page to high-resolution JPG
3. Saves to "pending/from production/images" with systematic naming
4. Names files as x1_1.jpg, x1_2.jpg, x2_1.jpg, etc.

Requirements:
- pip install pdf2image pillow
- Poppler for Windows (for pdf2image)
"""

import os
import glob
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import re


def natural_sort_key(text):
    """Sort strings with numbers naturally (x1_1, x1_2, x2_1, etc.)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]


def get_next_pdf_number():
    """Get the next available PDF number based on existing images."""
    pending_folder = "pending/from production/images"
    if not os.path.exists(pending_folder):
        return 1
    
    # Find existing x*.jpg files
    existing_files = glob.glob(os.path.join(pending_folder, "x*.jpg"))
    
    if not existing_files:
        return 1
    
    # Extract numbers from filenames
    numbers = []
    for file in existing_files:
        filename = Path(file).stem  # e.g., "x1_1", "x2_3"
        match = re.match(r'x(\d+)', filename)
        if match:
            numbers.append(int(match.group(1)))
    
    if not numbers:
        return 1
    
    return max(numbers) + 1


def convert_pdf_to_images(pdf_path, output_folder, pdf_number, dpi=300):
    """
    Convert a PDF file to high-resolution JPG images.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Output folder for images
        pdf_number: Number for this PDF (x1, x2, etc.)
        dpi: Resolution in DPI (default 300 for high quality)
    """
    try:
        print(f"Converting: {Path(pdf_path).name}")
        
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt='JPEG',
            thread_count=4  # Use multiple threads for faster processing
        )
        
        print(f"  Pages found: {len(images)}")
        
        # Save each page as JPG
        for page_num, image in enumerate(images, 1):
            # Create filename: x1_1.jpg, x1_2.jpg, etc.
            filename = f"x{pdf_number}_{page_num}.jpg"
            output_path = os.path.join(output_folder, filename)
            
            # Save with high quality
            image.save(output_path, 'JPEG', quality=95, optimize=True)
            
            print(f"  Saved: {filename} ({image.size[0]}x{image.size[1]} pixels)")
        
        return len(images)
        
    except Exception as e:
        print(f"❌ Error converting {Path(pdf_path).name}: {e}")
        return 0


def process_pdf_folder(input_folder="Production drawings (pdfs)", output_folder="pending/from production/images"):
    """
    Process all PDF files in the input folder.
    
    Args:
        input_folder: Folder containing PDF files
        output_folder: Output folder for JPG images
    """
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        print("Please create the folder and add your PDF files.")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {input_folder}")
        return
    
    # Sort PDF files naturally
    pdf_files.sort(key=lambda x: natural_sort_key(Path(x).name))
    
    print(f"=== PDF to Image Conversion ===")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"PDF files found: {len(pdf_files)}")
    print()
    
    # Get starting PDF number
    start_pdf_num = get_next_pdf_number()
    
    # Process each PDF
    total_pages = 0
    successful_pdfs = 0
    failed_pdfs = 0
    
    for i, pdf_file in enumerate(pdf_files):
        pdf_number = start_pdf_num + i
        pages_converted = convert_pdf_to_images(pdf_file, output_folder, pdf_number)
        
        if pages_converted > 0:
            successful_pdfs += 1
            total_pages += pages_converted
        else:
            failed_pdfs += 1
        
        print()
    
    # Summary
    print(f"=== Conversion Complete ===")
    print(f"Successful PDFs: {successful_pdfs}")
    print(f"Failed PDFs: {failed_pdfs}")
    print(f"Total pages converted: {total_pages}")
    print(f"Output folder: {output_folder}")
    
    # List created files
    if successful_pdfs > 0:
        print(f"\nCreated files:")
        image_files = glob.glob(os.path.join(output_folder, "x*.jpg"))
        image_files.sort(key=lambda x: natural_sort_key(Path(x).name))
        
        for img_file in image_files:
            print(f"  {Path(img_file).name}")


def main():
    """Main function with user interface."""
    print("=== PDF to Image Converter ===")
    print("Converts PDF pages to high-resolution JPG files")
    print()
    
    # Check if input folder exists
    input_folder = "Production drawings (pdfs)"
    if not os.path.exists(input_folder):
        print(f"❌ Folder '{input_folder}' not found.")
        print("Please create this folder and add your PDF files.")
        return
    
    # Get DPI from user
    try:
        dpi = int(input("Enter DPI for conversion (default 300): ") or "300")
        if dpi < 72:
            print("Warning: DPI below 72 may result in poor quality images.")
    except ValueError:
        dpi = 300
        print("Using default DPI: 300")
    
    print()
    
    # Process the files
    process_pdf_folder(input_folder, "pending/from production/images")


if __name__ == "__main__":
    main() 