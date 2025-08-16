import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from django.conf import settings
import math


def draw_square(messageData, data_input):


    print("Input:", data_input)

    width = 5  # Default
    match = re.search(r"(side|width)\s*(?:is\s*)?(\d+)", data_input, re.IGNORECASE)
    if match:
        width = int(match.group(2))

    area = width * width
    perimeter = 4 * width

    step1 = "Area = side × side"
    step2 = f"{width} × {width}"
    step3 = f"Area = {area} cm²"

    fig, ax = plt.subplots()
    square = patches.Rectangle((0, 0), width, width, linewidth=2, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(square)
    ax.set_xlim(0, width + 2)
    ax.set_ylim(0, width + 2)
    ax.set_title(f'Square ({width}cm × {width}cm)')
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()

    filename = 'square_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)
    image_url = settings.MEDIA_URL + filename

    if "perimeter" in data_input.lower():
        perimeter_message = (
            f"Perimeter of a square is the total length around all four sides.\n"
            f"Formula: 4 × side = 4 × {width} = {perimeter} cm"
        )

        messageData.append({
            "image_url": image_url,
            "last_Data": f"🟦 Square with side {width}cm.\nArea: {area} cm²\n{perimeter_message}",
            "solution": {
                "equation": "Area = side × side",
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "answer": f"The area of the square is {area} cm²",
                "perimeter": f"Perimeter = 4 × {width} = {perimeter} cm"
            },
            "message_info": f"A square with side {width}cm was drawn, with area and perimeter calculated."
        })
    else:
        messageData.append({
            "image_url": image_url,
            "message": f"This is a square with side {width}cm. The area is calculated below.",
            "solution": {
                "equation": "Area = side × side",
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "answer": f"The area of the square is {area} cm²"
            },
            "last_Data": f"🟦 A square with side {width}cm has been drawn and solved."
        })

    return messageData




def draw_triangle(messageData, data_input):

    # Default dimensions
    base = 10
    height = 10
    print("Input:", data_input)
    # Extract base and height from input
    match = re.search(r"base\s*(\d+)\D*height\s*(\d+)", data_input, re.IGNORECASE)
    print("dataMacthh",match)


    if match:
        base = int(match.group(1))
        height = int(match.group(2))

    area = 0.5 * base * height
    perimeter = base + height + int((base**2 + height**2)**0.5)  # Assuming right-angled triangle

    step1 = "Area = (base × height) ÷ 2"
    step2 = f"({base} × {height}) ÷ 2"
    step3 = f"Area = {area} cm²"

    fig, ax = plt.subplots()
    triangle = patches.Polygon([[0, 0], [base, 0], [0, height]],
                               closed=True, edgecolor='purple',
                               facecolor='lightgrey', linewidth=2)
    ax.add_patch(triangle)
    ax.set_title(f'Triangle (Base {base}cm, Height {height}cm)', fontsize=14)
    ax.set_xlim(0, base + 2)
    ax.set_ylim(0, height + 2)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()

    filename = 'triangle_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)

    image_url = settings.MEDIA_URL + filename

    solution = {
        "equation": "Area = (base × height) ÷ 2",
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "answer": f"Area = {area} cm², Perimeter ≈ {perimeter} cm",
        "perimeter": f"Perimeter = base + height + hypotenuse ≈ {base} + {height} + √({base}² + {height}²)"
    }

    messageData.append({
        "image_url": image_url,
        "message": f"🔺 Triangle with base {base}cm and height {height}cm.\nArea = {area} cm²\nPerimeter ≈ {perimeter} cm",
        "solution": solution,
        "message_info": f"A triangle was drawn and solved using base {base}cm and height {height}cm."
    })

    return messageData

def draw_rectangle(messageData, data_input):


    # Default values
    width, height = 10, 5

    # Try to extract both width and height
    match = re.search(r"width\s*(\d+)\D*height\s*(\d+)", data_input, re.IGNORECASE)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))

    area = width * height
    perimeter = 2 * (width + height)

    step1 = "Area = width × height"
    step2 = f"{width} × {height}"
    step3 = f"Area = {area} cm²"

    fig, ax = plt.subplots()
    rectangle = patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='purple', facecolor='lightgrey')
    ax.add_patch(rectangle)
    ax.set_xlim(0, width + 2)
    ax.set_ylim(0, height + 2)
    ax.set_title(f'Rectangle ({width}cm × {height}cm)', fontsize=14)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()

    filename = 'rectangle_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)

    image_url = settings.MEDIA_URL + filename

    solution = {
        "equation": "Area = width × height",
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "answer": f"Area = {area} cm² and Perimeter = {perimeter} cm",
        "perimeter": f"Perimeter = 2 × (width + height) = 2 × ({width} + {height}) = {perimeter} cm"
    }

    message = (
        f"This is a rectangle with width {width}cm and height {height}cm.\n"
        f"Area = {area} cm²\n"
        f"Perimeter = {perimeter} cm"
    )

    messageData.append({
        "image_url": image_url,
        "message": message,
        "solution": solution,
        "message_info": f"A rectangle with width {width}cm and height {height}cm was drawn."
    })

    return messageData

def draw_circle(messageData, data_input):
    radius = 5  # default radius

    # Extract radius from user input
    match = re.search(r"(radius|r)\s*(?:is\s*)?(\d+)", data_input, re.IGNORECASE)
    if match:
        radius = int(match.group(2))

    area = math.pi * radius ** 2
    perimeter = 2 * math.pi * radius

    step1 = "Area = π × radius²"
    step2 = f"3.14159× {radius}²"
    step3 = f"Area ≈ {area:.2f} cm²"

    step_perimeter = f"Perimeter = 2 × π × {radius} ≈ {perimeter:.2f} cm"

    # Plotting the circle
    fig, ax = plt.subplots()
    circle = patches.Circle((radius, radius), radius, edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(0, radius * 2 + 2)
    ax.set_ylim(0, radius * 2 + 2)
    ax.set_title(f'Circle (radius = {radius}cm)')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    filename = 'circle_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)

    image_url = settings.MEDIA_URL + filename

    solution = {
        "equation": "Area = π × radius²",
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4": "Perimeter = 2 × 3.14159 × radius",
        "step5": step_perimeter,
        "answer": f"Area ≈ {area:.2f} cm², Perimeter ≈ {perimeter:.2f} cm"
    }

    messageData.append({
        "image_url": image_url,
        "message": f"This is a circle with radius {radius}cm. Below are the area and perimeter calculations.",
        "solution": solution,
        "message_info": f"A circle with radius {radius}cm has been drawn, and both area and perimeter were calculated."
    })

    return messageData