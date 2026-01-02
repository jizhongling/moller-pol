#!/usr/bin/env python3
"""
Add clickable links to page numbers in the Table of Contents of a PDF file.
This script parses the TOC page and adds link annotations to make page numbers clickable.
"""

import sys
import re
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject, DictionaryObject, NameObject, NumberObject, ArrayObject, TextStringObject


def add_toc_links(input_pdf, toc_page_index=0):
    """
    Add clickable links to page numbers in the TOC.
    
    Args:
        input_pdf: Path to input PDF file
        toc_page_index: 0-based index of the TOC page (default: 0 for first page)
    """
    try:
        reader = PdfReader(input_pdf, strict=False)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        print("The PDF file may be corrupted or incomplete.")
        print("Please regenerate it using: root -l -b -q 'DrawWaveform.C()'")
        sys.exit(1)
    
    writer = PdfWriter()
    
    # Copy all pages to writer
    for page in reader.pages:
        writer.add_page(page)
    
    # Extract text from TOC page to find label-page mappings
    toc_page = reader.pages[toc_page_index]
    toc_text = toc_page.extract_text()
    
    # Parse TOC to find "Label" and "Start Page" entries
    # Expected format: "Label           Start Page"
    # followed by lines like "     5                    15"
    # OR "Start Page      6                      2" (first entry on same line as header)
    lines = toc_text.split('\n')
    
    label_page_map = []
    for line in lines:
        # Match lines with two numbers (label and page number)
        # Can appear at start or after text like "Start Page"
        matches = re.findall(r'(\d+)\s+(\d+)', line)
        for match in matches:
            label = int(match[0])
            page_num = int(match[1])
            label_page_map.append((label, page_num))
    
    label_count = len(label_page_map)
    print(f"Found {label_count} TOC entries:")
    for label, page in label_page_map:
        print(f"  Label {label} -> Page {page}")
    
    # Get page dimensions
    page_height = float(toc_page.mediabox.height)
    page_width = float(toc_page.mediabox.width)
    
    # Calculate approximate position for each entry
    # The TOC header "Start Page" is at approximately x=0.20, y=0.97 in NDC
    # Each entry line height is approximately 0.022 in text size
    # Convert NDC to page coordinates
    
    # Starting position for entries (below headers)
    start_x_ndc = 0.067 + 0.425 / label_count # X position for page numbers (top-down)
    page_num_width_ndc = 0.025  # Height to cover page numbers
    line_width_ndc = 0.85 / label_count  # Approximate spacing between lines
    
    # Page number column position
    page_num_y_ndc = 0.353  # Y position for first entry (left-right)
    page_num_height_ndc = 0.035  # Width to cover page numbers
    
    # Add link annotations for each page number
    for idx, (label, target_page) in enumerate(label_page_map):
        # Calculate x position for this entry (top-down)
        x_ndc = start_x_ndc + (idx * line_width_ndc)
        
        # Convert NDC to PDF coordinates (PDF origin is top-left)
        x0 = x_ndc * page_width
        y0 = page_num_y_ndc * page_height
        x1 = x0 + (page_num_width_ndc * page_width)
        y1 = y0 + (page_num_height_ndc * page_height)
        
        # Create link annotation
        link_rect = RectangleObject([x0, y0, x1, y1])
        
        # Target page index (0-based, so subtract 1 from page number)
        target_page_index = target_page - 1
        
        # Create the link annotation dictionary
        link_annotation = DictionaryObject()
        link_annotation.update({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Rect"): link_rect,
            NameObject("/Border"): ArrayObject([NumberObject(1), NumberObject(1), NumberObject(1)]),  # Visible red border
            NameObject("/C"): ArrayObject([NumberObject(1), NumberObject(0), NumberObject(0)]),  # Red color
            NameObject("/Dest"): ArrayObject([
                writer.pages[target_page_index].indirect_reference,
                NameObject("/XYZ"),
                NumberObject(0),
                NumberObject(page_height),
                NumberObject(0)
            ])
        })
        
        print(f"  Adding link box at x=[{x0:.1f}, {x1:.1f}], y=[{y0:.1f}, {y1:.1f}] for label {label} -> page {target_page}")
        
        # Add annotation to TOC page
        if "/Annots" in writer.pages[toc_page_index]:
            writer.pages[toc_page_index]["/Annots"].append(writer._add_object(link_annotation))
        else:
            writer.pages[toc_page_index][NameObject("/Annots")] = ArrayObject([writer._add_object(link_annotation)])
    
    # Write output
    output_pdf = input_pdf.replace('.pdf', '-linked.pdf')
    with open(output_pdf, 'wb') as f:
        writer.write(f)
    
    print(f"\nCreated {output_pdf} with {label_count} clickable links")
    
    # Replace original file
    import shutil
    shutil.move(output_pdf, input_pdf)
    print(f"Replaced {input_pdf} with linked version")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_toc_links.py <input_pdf> [toc_page_index]")
        print("  input_pdf: Path to PDF file with TOC")
        print("  toc_page_index: 0-based index of TOC page (default: 0)")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    toc_page_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    add_toc_links(input_pdf, toc_page_index)
