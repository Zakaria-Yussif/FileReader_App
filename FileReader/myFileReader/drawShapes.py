import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from django.conf import settings


def draw_square(messageData):

    fig, ax = plt.subplots()
    width = 5
    height = 5
    area= width*height

    message = (f"This is a square with base {width}cm and height {height}cm. "
               f"You can see it in the image below. Find the area:\n")

    step1 = f"Area= base * height "
    step2 = f"({width} * {height}) "
    step3 = f"Area = {area} cm²"
    solution=f"The area of a square is {area}cm²"
    message_info = (
        f"This is a square with side length {width}cm. "
        f"You can write a word problem involving a square, and I will draw and solve it!"
    )

    square = patches.Polygon([[0, 0], [width, 0], [width, width], [0, width]],
                             closed=True, edgecolor='blue',
                             facecolor='lightblue', linewidth=2)
    ax.add_patch(square)
    ax.set_xlim( width)
    ax.set_ylim(width)
    ax.set_title('Square (5x5)', fontsize=14)
    ax.set_xlabel('Base (5 cm)', fontsize=10)
    ax.set_ylabel('Height (5 cm)', fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    ax.set_aspect("equal")
    plt.tight_layout()

    filename = 'square_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)

    solution={
        "equation":"Area of a square= width * height ",
        "step1":step1,
        "step2":step2,
        "step3":step3,
        "answer":solution,

    }
    # Return image URL for frontend use
    image_url = settings.MEDIA_URL + filename

    messageData.append({
        "image_url": image_url,
        "message": message,
        "solution":solution,
        "message_info": message_info,
    })

    return messageData


def draw_triangle(messageData):
    fig, ax = plt.subplots()
    base = 10  # cm
    height = 10  # cm
    Area = (base * height) * 0.5  # Calculate area for a triangle

    # Message with area calculation for triangle
    message = (f"This is a triangle with base {base}cm and height {height}cm. "
               f"You can see it in the image below. Find the area:\n")

    step1=f"Area= base * height / 2"
    step2=f"({base} * {height}) / 2"
    step3=f"Area = {Area} cm²"


    triangle = patches.Polygon([[0, 0], [base, 0], [0, height]],
                               closed=True, edgecolor='purple',
                               facecolor='lightgrey', linewidth=2)
    ax.add_patch(triangle)
    ax.set_title('Triangle (Base 5x10)', fontsize=14)
    ax.set_xlabel('Base (5 cm)', fontsize=10)
    ax.set_ylabel('Height (10 cm)', fontsize=10)
    ax.set_xlim( height )
    ax.set_ylim( base)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

    filename = 'triangle_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)

    image_url = settings.MEDIA_URL + filename
    solution={
        "equation":"Area= base * height / 2",
        "step1":step1,
        "step2":step2,
        "step3":step3,
        "answer":f"Area of the triangle is {Area}cm²",



    }
    messageData.append({
        "image_url": image_url,
        "message": message,
        "solution":solution,
        "message_info": f"This is a triangle with side height {height}cm and base {base}cm. You can write a word problem involving a triangle, and I will draw and solve it!"
    })

    return messageData

def draw_rectangle(messageData):
    fig, ax = plt.subplots()
    width = 10
    height = 5
    area = width * height

    step1=f"Area= width* height"
    step2=f"({width} * {height})"
    step3= f"{area}cm²"
    solution=f"Area of rectangle= {area}cm²"

    message = (f"This is a rectangle with height {height}cm and width {width}cm:\n "
               f"You can see it in the image below. Find the area:\n")



    rectangle = patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='purple',facecolor='lightgrey' )
    ax.add_patch(rectangle)

    # Set limits to fit the rectangle
    ax.set_xlim(0, 7)  # a bit more space for margin
    ax.set_ylim(0, 3)

    # Title and axis labels
    ax.set_title('Rectangle (5x10)', fontsize=14)
    ax.set_xlabel('Width (5 cm)', fontsize=10)
    ax.set_ylabel('Height (10 cm)', fontsize=10)

    # Center text
    # ax.text(2.5, 5, 'Center', ha='center', va='center', fontsize=12, color='black')

    # Format: hide ticks but keep labels
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_aspect('equal')
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    # plt.show()

    filename = 'rectangle_plot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close(fig)

    # Return image URL for frontend use
    image_url = settings.MEDIA_URL + filename
    solution={
        "equation":"Area= width* height",
         "step1":step1,
        "step2":step2,
        "step3":step3,
        "answer":solution,


    }


    messageData.append({
        "image_url": image_url,
        "message": message,
        "solution":solution,
        "message_info": f"This is a rectangle with height {height}cm and width {width}cm. You can write a word problem involving a rectangle, and I will draw and solve it!"

    })

    return messageData
