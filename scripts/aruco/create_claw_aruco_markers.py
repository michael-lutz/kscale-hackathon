import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import os
import tempfile


def create_aruco_markers():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pdf_path = "aruco_claw_markers.pdf"
    c = canvas.Canvas(pdf_path, pagesize=(210 * mm, 297 * mm))  # A4 size

    # Add padding to the edges
    padding = 80 * mm
    marker_size_mm = 20 * mm  # Size in mm for PDF
    marker_size_px = 200  # Size in pixels for OpenCV

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(10):
            # Create ArUco marker
            marker = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
            cv2.aruco.generateImageMarker(aruco_dict, i, marker_size_px, marker, 1)

            # Save marker as temporary image file
            temp_image_path = os.path.join(tmpdirname, f"marker_{i}.png")
            cv2.imwrite(temp_image_path, marker)

            # Calculate position
            x = (i % 2) * (marker_size_mm + 10 * mm) + padding
            y = (4 - i // 2) * (marker_size_mm + 15 * mm) + padding

            # Draw marker on PDF
            c.drawImage(temp_image_path, x, y, width=marker_size_mm, height=marker_size_mm)

            # Add marker ID
            c.setFont("Helvetica", 6)
            c.drawString(x, y - 8 * mm, f"Marker ID: {i}")

    # Save PDF
    c.save()
    print(f"ArUco markers saved to {pdf_path}")


if __name__ == "__main__":
    create_aruco_markers()
