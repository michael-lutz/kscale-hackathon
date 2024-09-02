from reportlab.lib.pagesizes import landscape, letter  # type: ignore
from reportlab.lib.units import inch  # type: ignore
from reportlab.pdfgen import canvas  # type: ignore


def generate_chessboard_pdf(filename="chessboard.pdf", rows=6, cols=9, square_size=1.0):
    """Generates a chessboard pattern PDF for camera calibration.

    Args:
        filename (str): Name of the PDF file to save.
        rows (int): Number of squares in the vertical direction.
        cols (int): Number of squares in the horizontal direction.
        square_size (float): Size of each square in inches.
    """
    c = canvas.Canvas(filename, pagesize=landscape(letter))  # Rotate to landscape

    # Convert square size to points (1 inch = 72 points)
    square_size_points = square_size * inch

    # Set starting point for drawing (center the chessboard on the page)
    page_width, page_height = landscape(letter)
    start_x = (page_width - (cols * square_size_points)) / 2
    start_y = (page_height - (rows * square_size_points)) / 2

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                x = start_x + j * square_size_points
                y = start_y + i * square_size_points
                c.rect(x, y, square_size_points, square_size_points, fill=1)

    c.showPage()
    c.save()
    print(f"Chessboard PDF saved as '{filename}'.")


if __name__ == "__main__":
    filename = "chessboard.pdf"  # Output PDF file name
    rows = 6  # Number of rows of squares
    cols = 9  # Number of columns of squares
    square_size = 1.0  # Size of each square in inches

    generate_chessboard_pdf(filename, rows, cols, square_size)
