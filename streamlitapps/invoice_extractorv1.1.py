
import streamlit as st
import pytesseract
from PIL import Image
import PyPDF2
import io

import pdf2image


def ocr_pdf(file):
    """
    Extracts text and images from a PDF file, performing OCR on images.

    Args:
        file: A file object containing the PDF data.

    Returns:
        A tuple containing:
            - Extracted text from the PDF
            - A list of PIL Images, one for each page
    """

    text = ""
    images = []

    pdf_reader = PyPDF2.PdfReader(file)

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]

        # Extract text from the page (may contain structured data)
        extracted_text = page.extract_text()
        text += extracted_text

        pdf_bytes = io.BytesIO()
        pdf_writer = PyPDF2.PdfWriter()

        pdf_writer.add_page(page)
        pdf_writer.write(pdf_bytes)

        pdf_bytes.seek(0)

        # Extract image from the page (for better OCR of complex layouts)
        images.append(pdf2image.convert_from_bytes(pdf_bytes.getvalue())[0])

    return text, images


def extract_specific_information(text):
    """
    Extracts specific information (email, date, names, places, invoice details)
    from the OCRed text.

    Args:
        text: The OCRed text string.

    Returns:
        A dictionary containing the extracted information with keys:
            - email
            - date
            - names (list)
            - places (list)
            - invoice_number
            - order_number
            - invoice_date
            - due_date
    """

    extracted_info = {
        "email": None,
        "date": None,
        "names": [],
        "places": [],
        "invoice_number": None,
        "order_number": None,
        "invoice_date": None,
        "due_date": None,
    }

    # Use regular expressions or string parsing techniques to extract specific data
    # (Adapt these patterns to match your expected invoice format)
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    date_pattern = r"\d{1,2}/\d{1,2}/\d{4}"  # MM/DD/YYYY format
    invoice_number_pattern = r"INV-\d+"
    order_number_pattern = r"Order Number: \d+"
    invoice_date_pattern = r"Invoice Date: (.*)"
    due_date_pattern = r"Due Date: (.*)"

    for line in text.splitlines():
        email_match = re.search(email_pattern, line)
        if email_match:
            extracted_info["email"] = email_match.group(0)

        date_match = re.search(date_pattern, line)
        if date_match:
            extracted_info["date"] = date_match.group(0)

        # Extract names and places using appropriate patterns
        # (You might need additional logic here)
        if "Name" in line:  # Replace with relevant patterns
            extracted_info["names"].append(line.split(":")[1].strip())
        if "Place" in line:  # Replace with relevant patterns
            extracted_info["places"].append(line.split(":")[1].strip())

        invoice_number_match = re.search(invoice_number_pattern, line)
        if invoice_number_match:
            extracted_info["invoice_number"] = invoice_number_match.group(0)

        order_number_match = re.search(order_number_pattern, line)
        if order_number_match:
            extracted_info["order_number"] = order_number_match.group(0).split(": ")[-1]

        invoice_date_match = re.search(invoice_date_pattern, line)
        if invoice_date_match:
            extracted_info["invoice_date"] = invoice_date_match.group(1).strip()  # Extract captured date

        due_date_match = re.search(due_date_pattern, line)
        if due_date_match:
            extracted_info["due_date"] = due_date_match.group(1).strip()  # Extract captured date

    return extracted_info