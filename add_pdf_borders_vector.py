"""
add_pdf_borders_vector.py
-------------------------
Add simple rectangle borders to PDF pages while preserving all original content.
This creates vector PDFs with just a border rectangle added around each page.

Requirements:
- pip install PyPDF2 reportlab
"""

import os
import glob
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, red
import io


def add_border_to_pdf_vector(input_path, output_path, border_width=2):
    """
    Add a simple rectangle border to all pages of a PDF file.
    Preserves all original content as vector graphics.
    
    Args:
        input_path: Path to input PDF
        output_path: Path for output PDF
        border_width: Border thickness in points
    """
    try:
        print(f"Processing: {Path(input_path).name}")
        
        # Check if output file exists and is locked
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"  Removed existing file: {Path(output_path).name}")
            except PermissionError:
                print(f"  Error: Cannot overwrite {Path(output_path).name} - file may be open")
                return False
        
        # Read the input PDF
        reader = PdfReader(input_path)
        writer = PdfWriter()
        
        print(f"  Pages: {len(reader.pages)}")
        
        for page_num, page in enumerate(reader.pages):
            try:
                # Get page dimensions
                media_box = page.mediabox
                width = float(media_box.width)
                height = float(media_box.height)
                
                print(f"  Page {page_num + 1}: {width:.1f} x {height:.1f} points")
                
                # Create a new PDF with border
                border_packet = io.BytesIO()
                c = canvas.Canvas(border_packet, pagesize=(width, height))
                
                # Draw a thick red border that should be clearly visible
                c.setStrokeColor(red)
                c.setLineWidth(border_width * 2)  # Make it thicker
                c.rect(border_width, border_width, width - 2*border_width, height - 2*border_width)
                
                c.save()
                border_packet.seek(0)
                
                # Create border PDF reader
                border_reader = PdfReader(border_packet)
                border_page = border_reader.pages[0]
                
                # Create a new page that combines original content with border
                new_page = page
                new_page.merge_page(border_page)
                
                # Add the combined page
                writer.add_page(new_page)
                
            except Exception as page_error:
                print(f"  Error processing page {page_num + 1}: {page_error}")
                writer.add_page(page)
        
        # Write the output PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"[OK] Saved: {Path(output_path).name}")
        print(f"  Note: Original content preserved with simple rectangle border")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error processing {Path(input_path).name}: {e}")
        return False


def process_pdf_folder(folder_path="Production drawings (pdfs)", border_width=2):
    """
    Process all PDF files in the specified folder.
    
    Args:
        folder_path: Path to folder containing PDF files
        border_width: Border thickness in points
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        print("Please create the folder and add your PDF files.")
        return
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDF files found in: {folder_path}")
        return
    
    print(f"=== PDF Border Processing (Vector) ===")
    print(f"Folder: {folder_path}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Border width: {border_width} points")
    print()
    
    # Create output folder
    output_folder = f"{folder_path}_bordered"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        filename = Path(pdf_file).name
        output_path = os.path.join(output_folder, f"{Path(filename).stem}_bordered.pdf")
        
        print(f"\nProcessing: {filename}")
        
        if add_border_to_pdf_vector(pdf_file, output_path, border_width):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n=== Processing Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output folder: {output_folder}")
    print(f"\nNote: Creates vector PDFs with simple rectangle borders.")
    print(f"All original content is preserved as vector graphics.")


def main():
    """Main function with user interface."""
    import sys
    
    print("=== PDF Border Adder (Vector) ===")
    print("Adds simple rectangle borders to PDF files")
    print("Preserves all original content as vector graphics")
    print()
    
    # Check if folder exists
    folder_path = "Production drawings (pdfs)"
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder '{folder_path}' not found.")
        print("Please create this folder and add your PDF files.")
        return
    
    # Get border width from command line argument or user input
    if len(sys.argv) > 1:
        try:
            border_width = float(sys.argv[1])
            print(f"Using border width: {border_width} points")
        except ValueError:
            print(f"Invalid border width: {sys.argv[1]}")
            return
    else:
        # Get border width from user
        try:
            border_width = float(input("Enter border width in points (default 2): ") or "2")
        except ValueError:
            border_width = 2
            print("Using default border width: 2 points")
    
    print()
    
    # Process the files
    process_pdf_folder(folder_path, border_width)


if __name__ == "__main__":
    main() 