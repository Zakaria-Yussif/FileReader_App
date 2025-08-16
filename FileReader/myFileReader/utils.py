from math import trunc

import requests

# from adodbapi.examples.xls_read import filename
from django.shortcuts import render, redirect

import matplotlib.pyplot as plt

import re

import math
# from adodbapi.examples.xls_read import filename
from transformers import pipeline


import pandas as pd
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from django.http import JsonResponse
from docx import Document
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PyPDF2 import PdfReader
import langcodes
from django.core.files import File
from googletrans import Translator

from django.conf import settings
import pdfplumber
import os
from pathlib import Path
from .forms import RegisterForm
import numpy as np

import spacy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sympy import symbols, Eq, solve, simplify, pretty, sympify

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
# from .model import PlotImage


# GOOGLE_API_KEY = settings.GOOGLE_API_KEY
# CSE_ID = settings.CSE_ID
GOOGLE_API_KEY="AIzaSyCsRN4Q09MowXjxEGsyADuJcFFTSYQDq-8"
CSE_ID="658e29a77c0f64601"

filename = r"C:\Users\zakar\PycharmFileExtractor\FileApp\runserver.XLS"
#
# fill_mask = pipeline("fill-mask", model="bert-base-uncased")
# nlp = spacy.load("en_core_web_sm")


BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SPACY_DIR  = BASE_DIR / "spacy"

BERT_DIR = MODELS_DIR / "bert-base-uncased"
BART_DIR = MODELS_DIR / "facebook-bart-large-cnn"
GPT2_DIR = MODELS_DIR / "distilgpt2"
SPACY_EN = SPACY_DIR / "en_core_web_sm"

# If the server has no internet, keep these ON
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        pass
    try:
        if SPACY_EN.exists():
            return spacy.load(str(SPACY_EN))
    except Exception:
        pass
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    print("[spaCy] Using blank('en') fallback (no NER).")
    return nlp

def load_pipeline(task, model_dir, hub_id):
    try:
        if Path(model_dir).exists():
            return pipeline(task, model=str(model_dir), tokenizer=str(model_dir), local_files_only=True)
        # If you DO have internet/cached models, flipping offline env vars off will allow this:
        return pipeline(task, model=hub_id)
    except Exception as e:
        print(f"[pipeline:{task}] unavailable: {e}")
        return None

# filename path (optional)
filename = Path(r"C:\Users\zakar\PycharmFileExtractor\FileApp\runserver.XLS")

# Load models safely
fill_mask = load_pipeline("fill-mask", BERT_DIR, "bert-base-uncased")
summarizer_pipeline = load_pipeline("summarization", BART_DIR, "facebook/bart-large-cnn")
nlp = load_spacy()
translator = Translator()


def generate_barChart(data_input, messageData, request):
    try:
        # Split input into category-value pairs
        data_pairs = data_input.split(',')
        categories = []
        values = []

        for pair in data_pairs:
            category, value = pair.split(':')
            categories.append(category.strip())  # Ensure it's a string
            values.append(int(value.strip()))

        print(categories, values)  # Debugging

        # Create bar chart
        plt.figure(figsize=(8, 6))
        plt.bar(categories, values, color='skyblue', width=0.5)
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.title('User Input Bar Chart')

        plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines
        plt.tight_layout()  # Fix layout issues

        # Ensure media directory exists
        image_path = os.path.join(settings.MEDIA_ROOT, 'user_bar_chart.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the chart as an image
        plt.savefig(image_path, format='png')
        plt.close()

        # Set image URL for rendering in the template
        image_url = settings.MEDIA_URL + 'user_bar_chart.png'
        with open('square_plot.png', 'rb') as f:
            django_file = File(f)
            # img = PlotImage(title='Square Plot')
            # img.image.save('square_plot.png', django_file, save=True)

        # Append message and image to messageData
        messageData.append({"image_url": image_url, "message": data_input})
        request.session['messageData'] = messageData

    except Exception as e:
        messageData.append({"message": f"Error processing input: {str(e)}"})

    return messageData

def normalize_equation(equation):
    # Add * between coefficient and variable, e.g., 4x -> 4*x
    return re.sub(r'(?<=\d)(?=x)', '*', equation)
def linear_equation(input_data, match_linear, messageData, request):
    try:
        # Extract slope and intercept from regex match
        m = float(match_linear.group(1)) if match_linear.group(1) else 1
        b = float(match_linear.group(2)) if match_linear.group(2) else 0

        # Plotting the linear function
        x_vals = np.linspace(-10, 10, 100)
        y_vals = m * x_vals + b

        # Solve the equation symbolically
        x = symbols('x')
        input_data2 = normalize_equation(input_data)
        left, right = input_data2.replace(' ', '').split('=')
        equation = Eq(eval(left), eval(right))
        solutions = solve(equation, x)

        right_val = sympify(right)
        a = right_val - b
        c = float(f"{a:.2f}")

        # Plot and save the image
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f"y = {m}x + {b}", color='red')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Linear Equation: y = {m}x + {b}')
        plt.grid(True)
        plt.legend()

        image_filename = 'linear_equation.png'
        image_path = os.path.join(settings.MEDIA_ROOT, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path, format='png')
        plt.close()

        # Save image to database
        with open(image_path, 'rb') as f:
            django_file = File(f)
            # plot_image = PlotImage(title=f'Linear Graph: y = {m}x + {b}')
            # plot_image.image.save(image_filename, django_file, save=True)

        image_url = settings.MEDIA_URL + image_filename

        # Generate steps
        step1 = f"🔢 Step 1: Original equation:\n   {pretty(equation)}"
        if m != 0:
            step2 = f"✏️ Step 2: Subtract {b} from both sides:\n   {m}x = {right} - {b} → {m}x = {c}"
            step3 = f"📐 Step 3: Divide by {m}:\n   x = {c} / {m}"
            step4 = f"✅ Final Answer: x = {solutions[0]}"
        elif b != 0:
            step2 = "❌ Step 2: No solution exists since m = 0 and b ≠ 0"
            step3 = ""
            step4 = "✅ Final Answer: No solution"
        else:
            step2 = "♾️ Step 2: Equation reduces to 0 = 0"
            step3 = ""
            step4 = "✅ Final Answer: Infinite solutions"

        content = {
            "equation": input_data,
            "step1": step1,
            "step2": step2,
            "step3": step3,
            "step4": step4,
            "answer": f"x = {solutions[0]}" if solutions else step4,
            "message": f"The equation is: {input_data}. The solution is: x = {solutions[0] if solutions else step4}. The graph below shows the linear equation.",
            "image_url": image_url,
        }

        messageData.append({ "solution": content })
        request.session['messageData'] = messageData

    except Exception as e:
        messageData.append({"message": f"❌ Invalid input: {str(e)}"})

    return messageData
def quadratic(data_input,match_quadratic, messageData, request):
    a = float(match_quadratic.group(1) or 1)
    b = float(match_quadratic.group(2) or 0)
    c = float(match_quadratic.group(3) or 0)
    # a2 = float(match_quadratic.group(4) or 1)  # Default 1 for x²
    # b2 = float(match_quadratic.group(5) or 0)
    # c2 = float(match_quadratic.group(6) or 0)

    # Generate x values with higher resolution for smooth curves
    x = np.linspace(-10, 10, 400)

    # Compute y values for both equations
    y = a * x ** 2 + b * x + c
    # y2 = a2 * x ** 2 + b2 * x + c2

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"y = {a}x² + {b}x + {c}", color='skyblue', linewidth=2)
    # plt.plot(x, y2, label=f"y = {a2}x² + {b2}x + {c2}", color='darkred', linestyle='dashed', linewidth=2)

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Quadratic Equations: {a}x² + {b}x + {c} ', color='darkred')

    # Draw x and y axes
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Enable grid on both axes
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save the plot
    image_path = os.path.join(settings.MEDIA_ROOT, 'quadratic_plot.png')
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png')
    plt.close()

    # Return the image URL
    image_url = settings.MEDIA_URL + 'quadratic_plot.png'
    # PlotImage.objects.create(title="quadratic_plot.png:{a}x² + {b}x + {c}", image=image_url)
    content = {}

    if a != 0:
        try:
            discriminant = b ** 2 - 4 * a * c
            root1 = (-b + math.sqrt(discriminant)) / (2 * a)
            root2 = (-b - math.sqrt(discriminant)) / (2 * a)
            step1 = f"Step1:Calculate The descriminant: Δ= b² - 4 * a * c ",
            step2 = f"Step2:Δ= {b ** 2} - {4 * a * c}",
            step3 = f"Step3:Δ={discriminant}",
            if discriminant > 0:
                # root1 = (-b + math.sqrt(discriminant)) / (2 * a)
                # root2 = (-b - math.sqrt(discriminant)) / (2 * a)
                step1 = f"Step1: Calculate  The   descriminant: Δ= b² - 4 * a * c ",
                step2 = f"Step2:Δ= {b ** 2} - {4 * a * c}",
                step3 = f"Step3:Δ={discriminant}",
                step4 = "Step 4: Since Δ > 0, the equation has two real roots."
                step5 = f"Step 5:  formula:  x1 = (-b + √Δ) / (2a), x2 = (-b - √Δ) / (2a)"
                solution = f"Solution: x1 = {root1:.2f}, x2 = {root2:.2f}",


            elif discriminant == 0:
                # One real root (repeated)
                root = -b / (2 * a)
                step4 = "Step 4: Since Δ = 0, the equation has one real root (repeated)."
                step5 = "Step 5: Compute root: x = -b / (2a)"
                solution = f"Solution: x = {root:.2f}"

            else:
                # Complex roots
                root1 = (-b + complex(math.sqrt(abs(discriminant)))) / (2 * a)
                root2 = (-b - complex(math.sqrt(abs(discriminant)))) / (2 * a)
                step4 = "Step 4: Since Δ < 0, the equation has two complex roots."
                step5 = f"Step 5: Compute complex roots using formula: x1 = (-b + √Δ) / (2a), x2 = (-b - √Δ) / (2a)"
                solution = f"Solution: x1 = {root1:.2f}, x2 = {root2:.2f}"

            content = {
                "equation": f"{data_input} + '' + where a={a} , b={b} , c={c}",
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "step4": step4,
                "step5": step5,
                "image_url": image_url,
                "answer": solution,
                "message": f"The equation is: {data_input} and the solution is: x1 = {root1:.2f}, x2 = {root2:.2f} .The graph below shows the quadratic equation and the solution.",
                "image_url": image_url,

            }
            messageData.append({"image_url": image_url, "solution": content})
            request.session['messageData'] = messageData

        except Exception as e:
            messageData.append({"message": f"Error processing input a cannot be 0: {str(e)}"})
        return messageData

def quadratic_two(data_input,match_quadratic_two, messageData, request):
    a = float(match_quadratic_two.group(1) or 1)
    b = float(match_quadratic_two.group(2) or 0)
    c = float(match_quadratic_two.group(3) or 0)
    a1 = float(match_quadratic_two.group(4) or 0)
    b1 = float(match_quadratic_two.group(5) or 0)
    c1 = float(match_quadratic_two.group(6) or 0)

    x = np.linspace(-10, 10, 400)
    y = a * x ** 2 + b * x + c
    y1 = a1 * x ** 2 + b1 * x + c1
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f" y= {a}x² + {b}x + {c}", color='skyblue', linewidth=2)
    plt.plot(x, y1, label=f"y={a1}x² + {b1}x + {c1}", color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Quadratic Equations: y = {a}x² + {b}x + {c} and y = {a1}x² + {b1}x + {c1}", color='darkred')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Enable grid on both axespy
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save the plot
    image_path = os.path.join(settings.MEDIA_ROOT, 'quadratic_plot.png')
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png')
    plt.close()

    # Return the image URL
    image_url = settings.MEDIA_URL + 'two_quadratic_plot.png'
    # PlotImage.objects.create(tittle="Two Quadratic Equations",image_url=image_url)

    if a != 0 or a1 != 0:
        try:
            discriminant = b ** 2 - 4 * a * c
            discriminant2 = (-b1 + discriminant) / (2 * a1)
            root1 = (-b + math.sqrt(discriminant)) / (2 * a)
            root2 = (-b - math.sqrt(discriminant)) / (2 * a)
            root3 = (-b1 + math.sqrt(discriminant)) / (2 * a1)
            root4 = (-b1 - math.sqrt(discriminant)) / (2 * a1)
            step1 = (f"Step1: Calculate The discriminant Δ = b² - 4 * a * c \n"
                     f"Discriminant2: Δ = b² - 4 * a * c")
            step2 = (f"Step2:Δ1= {b ** 2} - {4 * a * c}\n "
                     f" Δ2= {b1 ** 2} - {4 * a1 * c1} "),
            step3 = (f"Step3:Δ1={discriminant} \n"
                     f"Δ2= {discriminant2}"),
            if discriminant > 0:
                # root1 = (-b + math.sqrt(discriminant)) / (2 * a)
                # root2 = (-b - math.sqrt(discriminant)) / (2 * a)
                step1 = (f"Step1: Calculate  The   Δ1: Δ1= b² - 4 * a * c   \n"
                         f"Calculate Δ2:b1²-4 * a1 * c1   ")
                step2 = (f"Step2:Δ1= {b ** 2} - {4 * a * c}  \n"
                         f"Δ2= {b ** 2} - {4 * a * c} "),
                step3 = (f"Step3:Δ1={discriminant:.2f} \n "
                         f"Δ2={discriminant:.2f} "),
                step4 = "Step 4: Since Δ1 > 0 OR Δ2 > 0 the equation has two distinct  real roots."
                step5 = (f"Step 5: formula1:  x1 = (-b + √Δ) / (2a), x2 = (-b - √Δ) / (2a) \n "
                         f"formula2:  x2 = (-b1 + √Δ2) / (2a1), x2 = (-b1 - √Δ2) / (2a1) ")
                solution = (f"Solution: formula1: x1 = {root1:.2f}, x2 = {root2:.2f}   \n   "
                            f"formula2:  x2 = {root1:.2f}, x2= {root2:.2f}"),

            elif discriminant == 0:
                # One real root (repeated)
                root = -b / (2 * a)
                root2 = -b1 / (2 * a1)
                step4 = "Step 4: Since Δ1 = 0, AND Δ2=0 the equation has one real root (repeated)."
                step5 = ("Step 5: Compute root: x1 = -b / (2a)\n  "
                         f"x2=-b1/(2a1)")
                solution = (f"Solution: x1 = {root:.2f} \n"
                            f" x2={root2:.2f} ")

            else:
                # Complex roots
                root1 = (-b + complex(math.sqrt(abs(discriminant)))) / (2 * a)
                root3 = (-b1 - complex(math.sqrt(abs(discriminant)))) / (2 * a1)
                root2 = (-b - complex(math.sqrt(abs(discriminant)))) / (2 * a)
                root4 = (-b1 + complex(math.sqrt(abs(discriminant)))) / (2 * a1)
                step4 = "Step 4: Since Δ < 0, the equation has two complex roots."
                step5 = (f"Step 5: formula1:  x1 = (-b + √Δ) / (2a), x2 = (-b - √Δ) / (2a)\n "
                         f"formula2:  x2 = (-b1 + √Δ2) / (2a1), x2 = (-b1 - √Δ2) / (2a1) ")
                solution = (f"Solution: x1 = {root1:.2f} \n "
                            f"x2 = {root2:.2f}")

            content = {
                "equation": f"{data_input}  where a={a} , b={b} , c={c} AND  a={a1} , b={b1} , c={c1}",
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "step4": step4,
                "step5": step5,
                "answer": solution,
                "message": f"The equation is: {data_input} and the solution is: x1 = {root1:.2f}, x2 = {root2:.2f} .The graph below shows two quadratic equation and the solution.",
                "image_url": image_url,


            }

            messageData.append({"image_url": image_url, "solution": content})
            request.session["messageData"] = messageData
            return render(request, 'myFileApp/index.html', {'messageData': messageData})

        except Exception as e:
            messageData.append({"message": f"Error processing input a cannot be 0: {str(e)}"})
            return render(request, 'myFileApp/index.html', {'messageData': messageData})
        return messageData


def generate_essay(prompt):
    generator = pipeline("text-generation", model="distilgpt2")
    essay = generator(prompt, do_sample=True, temperature=0.7,truncation=True)
    return essay[0]["generated_text"]


ALLOWED_FILE_TYPES = [
    'application/pdf',               # PDF
    'image/png',                      # PNG Image
    'application/msword',             # DOC (Word document)
    'text/csv',                       # CSV
    'image/jpeg',                     # JPEG Image
    'text/plain'                      # TXT
]


data_samples = [
    # Format: (List of Named Entity Types, Label)
    (['MONEY', 'ORG', 'QUANTITY', 'CARDINAL', 'DATE'], 'Mathematical Exercise'),
    (['QUANTITY', 'CARDINAL', 'DATE'], 'Physics'),
    (['MONEY', 'ORG', 'CARDINAL'], 'Finance'),
    (['PERSON', 'ORG', 'DATE', 'CARDINAL'], 'General Text'),
    (['CARDINAL', 'ORG', 'QUANTITY', 'MONEY'], 'Economics'),
]


def extract_text_from_pdf(file_path):
    headers = []
    paragraphs = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = page_text.split('\n')  # Split the page text into lines
                for line in lines:
                    # Check if the line is in uppercase and treat it as a header
                    if line.isupper() and len(line.strip()) > 0:
                        parts = line.strip().split('.')
                        headers.append(parts[0])
                    else:
                        # Treat all other lines as paragraphs and split them into parts
                        # Split by punctuation marks (you can adjust the split logic)
                        parts = line.strip().split('.')  # This splits by periods
                        for part in parts:
                            if part.strip():  # Make sure the part is not empty
                                paragraphs.append(part.strip())
                                print({"paragraph":paragraphs})

    return headers, paragraphs


def extract_tables_from_pdf(file_path):
    tables = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)
    return tables


def extract_links_from_pdf(file_path):
    links = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        if '/Annots' in page:
            for annot in page['/Annots']:
                uri = annot.get_object().get('/A', {}).get('/URI', None)
                if uri:
                    links.append(uri)
    return links


def extract_bullets_from_pdf(file_path):
    bullets = []
    bullet_symbols = {'•', '-', '*', "numbers"}  # Set of bullet symbols
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()  # Extract text from each page
            if page_text:
                lines = page_text.split('\n')  # Split text into lines
                for line in lines:
                    # Check if the first non-whitespace character matches any bullet symbol
                    if line.lstrip()[0] in bullet_symbols:
                        bullets.append(line.strip())
                        # Remove leading/trailing whitespace
    return bullets




def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def entity_vector(entities):
    entity_counts = Counter(entities)
    return [
        entity_counts.get('MONEY', 0),
        entity_counts.get('ORG', 0),
        entity_counts.get('QUANTITY', 0),
        entity_counts.get('CARDINAL', 0),
        entity_counts.get('DATE', 0),
        entity_counts.get('PERSON', 0)
    ]

# Prepare training data
X = [entity_vector(sample[0]) for sample in data_samples]
y = [sample[1] for sample in data_samples]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

def extract_entities_from_text(text):
    doc = nlp(text)
    return [ent.label_ for ent in doc.ents]




def translate_text(data_input, target_language):
    # Initialize the Translator
    translator = Translator()

    # Translate the text to the target language
    translated = translator.translate(data_input, dest=target_language)




    return translated


def upload_files(uploaded_file, messageData,request):
    messageData = request.session.get("messageData", [])
    # Validate file type
    if uploaded_file.content_type not in ALLOWED_FILE_TYPES:
        return {"error": "File type not supported."}

    # Ensure 'uploads/' directory exists
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # Save the file to the server
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Initialize variables
        extracted_text, extracted_headers, extracted_paragraphs, extracted_tables = None, None, None, None

        # Extract content based on file type
        if uploaded_file.content_type == 'application/pdf':
            extracted_headers, extracted_paragraphs = extract_text_from_pdf(file_path)
            extracted_tables = extract_tables_from_pdf(file_path)
            extracted_bullets = extract_bullets_from_pdf(file_path)

            # extracted_headers_data = [word.strip(" []'' ").lower() for word in extracted_headers]  # if it's a list of words
            # headers = " ".join(extracted_headers_data)

            # extracted_tables_data=[word.strip(" []'' ").lower() for word in extracted_tables]
            # tables = " ".join(extracted_tables_data)



            content = {
                "mainHeader": uploaded_file.name,
                "title": extracted_headers,
                "paragraph": extracted_paragraphs,
                "bullets": extracted_bullets,
                "table": extracted_tables,
            }

            text = content["mainHeader"] + " " + " ".join(content["title"]) + " " + " ".join(content["paragraph"])
            summarise_data=summarise_message(text)
            detected_language = translator.detect(text).lang
            full_language_name = langcodes.Language.get(detected_language).language_name()
            print(f"Detected language: {full_language_name}")

            # Named Entity Recognition (NER)
            doc = nlp(text)
            name_entities = [(ent.text, ent.label_) for ent in doc.ents]
            entity_labels = [ent[1] for ent in name_entities]

            entity_features = entity_vector(entity_labels)
            predicted_category = model.predict([entity_features])[0]

            print(f"Predicted Document Type: {predicted_category}")

            # Create entity message
            entity_counts = Counter(entity_labels)
            entities_message = ", ".join([f"{label}: {count}" for label, count in entity_counts.items()])

            # Prepare the message for display
            if detected_language != "en":
                messageTrans = f"This document is categorized as {predicted_category}. It contains {len(name_entities)} entities such as {entities_message}."
                content["message"] = messageTrans
                messageData.append({"fileData": content,"message":summarise_data})

                request.session['messageData'] = messageData

                return messageData

            message = f"The file is written in {full_language_name}. Would you like to translate it into English?"
            content["message"] = message

                # Save original content
            messageData.append({"fileData": content})
            request.session['messageData'] = messageData
            return messageData

        elif uploaded_file.content_type == 'text/csv':
            extracted_text = read_csv(file_path)
        elif uploaded_file.content_type == 'application/msword':
            extracted_text = extract_text_from_docx(file_path)
        elif uploaded_file.content_type == 'text/plain':
            extracted_text = read_txt(file_path)

        return messageData

    except Exception as e:
        return {"error": f"Error saving file: {str(e)}"}


def detect_language(data_input):
    language = translator.detect(data_input).lang
    return language


#

# Load the model once (globally or inside a setup)
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarise_message(data_input, messageData, min_len=30, max_len=100):
    if not data_input or len(data_input.strip()) < 20:
        return "No sufficient content to summarize."

    try:
        # Dynamically adjust max and min lengths based on input size
        input_len = len(data_input.split())
        max_len = min(max_len, max(20, int(input_len * 0.6)))
        min_len = min(min_len, max(10, int(input_len * 0.4)))

        summary = summarizer_pipeline(data_input, max_length=max_len, min_length=min_len, do_sample=False)
        summarized_text = summary[0]['summary_text']

        print("summarization:", summarized_text)
        messageData.append({"message": summarized_text})

        return summarized_text, messageData
    except Exception as e:
        return f"Error during summarization: {e}"


def google_search(query, num_results=3):
    """Perform a Google search using the API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": CSE_ID,
        "num": num_results
    }

    response = requests.get(url, params=params)
    results = response.json()

    return results.get("items", [])  # Returns search results or empty list
