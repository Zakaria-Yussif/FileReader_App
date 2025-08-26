import base64
import fileinput
from cProfile import label
from copy import deepcopy
import pandas as pd
import numpy as np
import requests
import pycountry
from sympy import symbols, Eq, solve, simplify, pretty



import spacy
# from adodbapi.examples.xls_read import filename
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import matplotlib.pyplot as plt
import uuid

import re
import nltk
from rapidfuzz import process, fuzz
from nltk.chat.util import Chat, reflections
from IPython.display import display, Math
import cmath
import math
# from adodbapi.examples.xls_read import filename
from transformers import pipeline
from googletrans import Translator

from bs4 import BeautifulSoup
from googletrans import LANGUAGES
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer



from docx import Document
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PyPDF2 import PdfReader
# from .model import PlotImage
import langcodes
# from googletrans import Translator
from deep_translator import GoogleTranslator



# from .forms import RegisterForm  # Assuming RegisterForm is defined in your forms.py


# Home view
filename = r"C:\Users\zakar\PycharmFileExtractor\FileApp\runserver.XLS"
file_path = r"C:\Users\zakar\.cache\kagglehub\datasets\itachi9604\disease-symptom-description-dataset\versions\2\dataset.csv"
file_path_description=r"C:\Users\zakar\.cache\kagglehub\datasets\itachi9604\disease-symptom-description-dataset\versions\2\symptom_Description.csv"
file_path_precaution=r"C:\Users\zakar\.cache\kagglehub\datasets\itachi9604\disease-symptom-description-dataset\versions\2\symptom_precaution.csv"
# Train model and get encoders
file_path_blood_Results= r"C:\Users\zakar\PycharmFileExtractor\FileApp\diseases.data\blood_Results.csv"

print(file_path_blood_Results)
# Update the path as needed
translator = Translator()

messageInfo = ""


messageData = []
messageData2 = []

graphData = []
messageInfo = ""
Non_English = False
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

def index(request):
    """Render the homepage with stored messages"""

    default_message = (
        "Write Something OR upload a file \n"
        "You can upload pdf files, solve linear/quadratic equations and generate bar graphs"
    )

    # Retrieve messageData from session or set to empty list
    messageData = request.session.get('messageData', [])

    # Make sure it's a list
    if not isinstance(messageData, list):
        messageData = []

    # Append message only once
    if default_message not in messageData:
        messageData.append(default_message)

    # Save it back to session
    request.session['messageData'] = messageData

    print("Current session messageData:", messageData)

    return render(request, 'myFileReader/index.html', {'message': messageData})


# def read_excel_file():
#     print("Reading Excel File")


# def retrieved_data(request):
#     if PlotImage.objects.exists():
#         print("PlotImage.objects.exists()")
#     else:
#         print("PlotImage.donot.exists()")
   #     data= PlotImage.objects.all(image=request.FILES['image'])
   # print("retrieved_data", data)
   # graphData.append(data)
   # request.session['graphData'] = graphData
   # return render(request, 'myFileApp/index.html', {'graphData': graphData})

def submit_message(request):
    """Processes user input and generates responses (translation-first, robust returns)."""
    global messageData
    global messageInfo
    global complete
    global data_input
    global Non_English
    global detected_language

    from django.shortcuts import render
    from django.http import JsonResponse

    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Invalid method"}, status=405)

    # ---- 1) Read & normalize raw input ----
    raw = (request.POST.get('message', '') or '').strip()
    original_input = re.sub(r"\s+", " ", raw).strip()
    if not original_input:
        # Ensure messageData is a list
        messageData = request.session.get('messageData', [])
        if not isinstance(messageData, list):
            messageData = [messageData] if isinstance(messageData, dict) else []
        messageData.append({"message": "⚠️ Please enter a valid sentence!"})
        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # ---- 2) Ensure session messageData is a list ----
    messageData = request.session.get('messageData', [])
    if not isinstance(messageData, list):
        messageData = [messageData] if isinstance(messageData, dict) else []

    # ---- 3) Detect & translate: make input_data ENGLISH if not English ----
    # Rely on translator's .src for true language code; do NOT use a detect_language function here.
    from .utils import translate_text
    input_data = original_input        # canonical working text (English when translated)
    detected_language = "en"           # original language code, e.g. 'fr', 'ar', 'en'
    Non_English = False
    full_language_name = "English"

    try:
        tr = translate_text(original_input, target_language="en")
        src_code = (getattr(tr, "src", None) or "en").lower()
        if src_code != "en":
            Non_English = True
            detected_language = src_code
            input_data = getattr(tr, "text", None) or original_input
            try:
                full_language_name = langcodes.Language.get(src_code).language_name()
            except Exception:
                full_language_name = src_code
        else:
            detected_language = "en"
            input_data = original_input
            full_language_name = "English"
        print(f"[Lang] detected={detected_language} ({full_language_name})")
        if Non_English:
            print(f"[Lang] translated -> EN: {input_data}")
    except Exception as e:
        # Translation/detection failed: proceed with original
        Non_English = False
        detected_language = "en"
        input_data = original_input
        full_language_name = "English"
        print(f"[Lang] translation failed, using original. {e}")

    # ---- 4) Use English transparently for ALL downstream logic ----
    data_input = input_data  # your code below uses data_input everywhere

    # Optional: keep some context
    request.session["original_input"] = original_input
    request.session["input_data"] = input_data
    request.session["input_lang"] = detected_language
    request.session["input_lang_name"] = full_language_name

    # ---- 5) Process text with SpaCy + BERT fill-mask suggestion ----
    # Lowercase for NLP processing (keep original_input shown back to user)
    data_input = data_input.lower()
    doc = nlp(data_input)

    has_subject, has_verb, has_object = (False, False, False)
    missing_word_position = None
    tokens = [token.text for token in doc]

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            has_subject = True
        if token.pos_ in ("VERB", "AUX"):
            has_verb = True
        if token.dep_ in ("dobj", "attr", "ccomp", "xcomp"):
            has_object = True

    # Insert [MASK] where necessary
    if not has_subject:
        missing_word_position = 0
        tokens.insert(0, "[MASK]")
    elif not has_verb:
        for i, t in enumerate(doc):
            if t.pos_ == "NOUN" or t.dep_ == "nsubj":
                missing_word_position = i + 1
                tokens.insert(i + 1, "[MASK]")
                break
    elif not has_object:
        for i, t in enumerate(doc):
            if t.pos_ == "VERB":
                missing_word_position = i + 1
                tokens.insert(i + 1, "[MASK]")
                break

    if missing_word_position is None:
        tokens.append("[MASK]")
        missing_word_position = len(tokens) - 1

    masked_sentence = " ".join(tokens)
    masked_sentence = masked_sentence.replace("[MASK]", "", masked_sentence.count("[MASK]") - 1)

    if masked_sentence.count("[MASK]") != 1:
        messageData.append({"message": "⚠️ Error: More than one [MASK] token found."})
        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # Predict missing word
    try:
        prediction = fill_mask(masked_sentence)
        predicted_word = prediction[0]["token_str"].strip()
    except Exception as e:
        predicted_word = ""
        print("fill-mask failed:", e)
    if predicted_word:
        tokens[missing_word_position] = predicted_word
    complete_sentence = " ".join(tokens).strip()

    # Show the user's original message (not lowercased) in the chat history
    messageData.append({"message": original_input, "id": 123})
    request.session['messageData'] = messageData

    # ---- 6) Intent patterns & matches (operate on EN data_input) ----
    actions = ["draw", "generate", "show", "solve", "calculate", "compute", "if",
               "determine", "evaluate", "work out", "find"]
    shapes = ["triangle", "square", "circle", "rectangle", "star"]

    action_pattern = r"|".join([re.escape(a) for a in actions])
    shape_pattern = r"|".join([f"{s}s?" for s in shapes])
    draw_shape_regex = rf"\b(?:{action_pattern})\b.*?\b({shape_pattern})\b"

    easy_pattern = r"\b(write|generate|create|need).*?\bessay\b"
    translate_pattern = r"(?:please\s+)?(?:can you\s+)?translate\s+(.+?)\s+(?:to|in|into)\s+(\w+)"

    # bar data like "A:10, B:20"
    Bar_pattern = r"^\s*\w+\s*:\s*\d+\s*(,\s*\w+\s*:\s*\d+\s*)*$"
    # general linear eq
    linear_pattern = r"^([-+]?\d*\.?\d+)?\s*\*?\s*x\s*([-+]?\d*\.?\d+)?\s*\*?\s*y?\s*([-+]?\d+(\.\d+)?)?\s*=\s*([-+]?\d*\.?\d+)$"

    diseases_pattern = re.compile(
        r"""
        ^                             
        (?:.*?\b(?:have|got|experiencing|suffering\s+from)\b)
        \s*
        (?P<symptoms>
            (?:
                [^,;/\|\n\r]+?
            )
            (?:
                (?:[,;/\|\n\r]|
                 \band\b|
                 \bor\b|
                 \band/or\b)
                [^,;/\|\n\r]+?
            )*
        )
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE
    )

    summarise = r"\b(?:please\s*)?(?:can|could|would)?\s*(you\s*)?(please\s*)?(summarize|summarise)\b"
    quadratic_pattern = r"^([-+]?\d*\.?\d+|\d+)x\^2\s*([-+]?\d*\.?\d+|\d*)x\s*([-+]?\d*\.?\d+|\d+)?$"
    quadratic_pattern_two = r"^([-+]?\d*\.?\d+|\d+)\s*x\^2\s*([-+]?\d*\.?\d+|\d+)?\s*x\s*([-+]?\d*\.?\d+|\d+)?\s*(?:[:;,]\s*|\s*and\s*)\s*([-+]?\d*\.?\d+|\d+)\s*x\^2\s*([-+]?\d*\.?\d+|\d+)?\s*x\s*([-+]?\d*\.?\d+|\d+)?$"
    draw_images_patternData = r"(?:please\s*)?(?:can|could|would)?\s*(?:you\s*)?(?:please\s*)?(draw|show|generate|create)\s+(?:an?\s*)?(?:image|picture|photo|drawing)?\s*(?:of|about)?\s*(?P<description>.+)"

    match_summarise = re.search(summarise, data_input, re.IGNORECASE)
    match_essay = re.search(easy_pattern, data_input, re.IGNORECASE)
    Bar_match = re.search(Bar_pattern, data_input, re.IGNORECASE)
    match_translate_Data = re.search(translate_pattern, data_input, re.IGNORECASE)
    match_linear = re.search(linear_pattern, data_input, re.IGNORECASE)
    match_quadratic = re.search(quadratic_pattern, data_input, re.IGNORECASE)
    match_quadratic_two = re.search(quadratic_pattern_two, data_input, re.IGNORECASE)
    match_draw_imageData = re.search(draw_images_patternData, data_input.strip(), re.IGNORECASE)
    match_disease = re.findall(diseases_pattern, data_input)
    match_draw_shapes = re.search(draw_shape_regex, data_input, re.IGNORECASE)

    # ---- 7) Solve-by-pattern (math model quick path) ----
    pattern_solve = r"""^Solve for x: \d+x \+ \d+ = \d+,x = -?\d+$|
                        ^John has \d+ apples and buys \d+ more\. How many apples does he have\?,-?\d+$|
                        ^Find the derivative of \d+x\^\d+,-?\d+x\^\d+$|
                        ^Evaluate f\(x\) = x\^2 - \d+x \+ \d+ at x = -?\d+,f\(-?\d+\) = -?\d+$"""
    match_solve_partern = re.search(pattern_solve, data_input, re.IGNORECASE)

    if match_solve_partern:
        from .mathModel import predict
        sol = predict(question=data_input)
        content = {"equation": data_input, "step3": sol, "answer": sol}
        messageData.append({"solution": content})
        request.session['messageData'] = messageData

    # ---- 8) Essay generator ----
    if match_essay:
        messageData.append({"message": complete_sentence})
        from .utils import generate_essay
        essayData = generate_essay(data_input)
        messageData.append({"message": essayData})
        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # ---- 9) Image search (list links) ----
    if match_draw_imageData:
        from .draw_images import google_search_image
        description = match_draw_imageData.group("description").strip()
        query = description
        results = google_search_image(query)
        if isinstance(results, list):
            for index, result in enumerate(results):
                link = result.get("link") if isinstance(result, dict) else None
                if not link:
                    continue
                label = f"image{index + 1}:link"
                messageData.append({"image_url": link, "label": label})
            messageData.append({"last_Data": "Which of the following images would you like me to draw for you?"})
        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # ---- 10) Draw shapes ----
    if match_draw_shapes:
        shape = match_draw_shapes.group(1).lower().rstrip('s')
        if shape == "triangle":
            messageData.append({"message": complete_sentence})
            from .drawShapes import draw_triangle
            messageData = draw_triangle(messageData, data_input)
        elif shape == "square":
            messageData.append({"message": complete_sentence})
            from .drawShapes import draw_square

            # remember where new items will start
            start_len = len(messageData)

            # draw_square is expected to append to messageData and return it
            messageData = draw_square(messageData, data_input)

            # If original input wasn't English, translate ONLY the newly-added strings
            if Non_English and detected_language != "en":
                from .utils import translate_text

                def _tx(s: str) -> str:
                    try:
                        tr = translate_text(s,
                                            target_language=detected_language)  # detected_language is a CODE like 'fr'
                        return getattr(tr, "text", None) or s
                    except Exception:
                        return s

                # Translate strings in the items added by draw_square
                for i in range(start_len, len(messageData)):
                    item = messageData[i]
                    if isinstance(item, dict):
                        # translate top-level string fields
                        for k, v in list(item.items()):
                            if isinstance(v, str):
                                item[k] = _tx(v)
                            elif isinstance(v, dict):
                                # translate strings one level deep (e.g., solution subfields)
                                item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                            elif isinstance(v, list):
                                item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                    elif isinstance(item, str):
                        messageData[i] = _tx(item)

            request.session['messageData'] = messageData
            return render(request, 'myFileReader/index.html', {'messageData': messageData})

        elif shape == "rectangle":
            messageData.append({"message": complete_sentence})
            from .drawShapes import draw_rectangle
            messageData = draw_rectangle(messageData, data_input)
            start_len = len(messageData)
            if Non_English and detected_language != "en":
                from .utils import translate_text

                def _tx(s: str) -> str:
                    try:
                        tr = translate_text(s,
                                            target_language=detected_language)  # detected_language is a CODE like 'fr'
                        return getattr(tr, "text", None) or s
                    except Exception:
                        return s

                # Translate strings in the items added by draw_square
                for i in range(start_len, len(messageData)):
                    item = messageData[i]
                    if isinstance(item, dict):
                        # translate top-level string fields
                        for k, v in list(item.items()):
                            if isinstance(v, str):
                                item[k] = _tx(v)
                            elif isinstance(v, dict):
                                # translate strings one level deep (e.g., solution subfields)
                                item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                            elif isinstance(v, list):
                                item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                    elif isinstance(item, str):
                        messageData[i] = _tx(item)

            request.session['messageData'] = messageData
            return render(request, 'myFileReader/index.html', {'messageData': messageData})


        elif shape == "circle":
            from .drawShapes import draw_circle
            messageData = draw_circle(messageData, data_input)
            start_len = len(messageData)
            if Non_English and detected_language != "en":
                from .utils import translate_text

                def _tx(s: str) -> str:
                    try:
                        tr = translate_text(s,
                                            target_language=detected_language)  # detected_language is a CODE like 'fr'
                        return getattr(tr, "text", None) or s
                    except Exception:
                        return s

                # Translate strings in the items added by draw_square
                for i in range(start_len, len(messageData)):
                    item = messageData[i]
                    if isinstance(item, dict):
                        # translate top-level string fields
                        for k, v in list(item.items()):
                            if isinstance(v, str):
                                item[k] = _tx(v)
                            elif isinstance(v, dict):
                                # translate strings one level deep (e.g., solution subfields)
                                item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                            elif isinstance(v, list):
                                item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                    elif isinstance(item, str):
                        messageData[i] = _tx(item)

            request.session['messageData'] = messageData
            return render(request, 'myFileReader/index.html', {'messageData': messageData})



    # ---- 11) Translate-on-command ("translate ... into <lang>") ----
    if match_translate_Data:
        from .utils import translate_text
        cleaned_sentence = re.sub(
            r'\b(translate|can you|can|into|please|in|to)\b',
            '',
            data_input,
            flags=re.IGNORECASE
        ).strip()

        words = cleaned_sentence.lower().split()
        last_word = words[-1] if words else ''
        last_two_words = " ".join(words[-2:]) if len(words) >= 2 else last_word

        detected_target = None
        phrase = cleaned_sentence.lower()

        for code, name in LANGUAGES.items():
            name_lower = name.lower()
            if last_word == name_lower or last_two_words == name_lower or last_word == code.lower():
                detected_target = code
                if name_lower in phrase:
                    phrase = phrase.replace(name_lower, '').strip()
                elif code.lower() in phrase:
                    phrase = phrase.replace(code.lower(), '').strip()
                break

        if detected_target:
            tr_out = translate_text(phrase, target_language=detected_target)
            trans_message = {
                "user_input": original_input,
                "translator": f"This is the translated sentence into {LANGUAGES[detected_target]}:",
                "translator_text": getattr(tr_out, "text", None) or phrase
            }
            messageData.append({"translated": trans_message})
            request.session['messageData'] = messageData
            return render(request, 'myFileReader/index.html', {'messageData': messageData})
        else:
            # If user asked translation but we couldn't detect target, try translating last fileData to English as fallback
            messageData.append({"message": "This language is not supported yet. Please try again with a different language."})
            if len(messageData) >= 2:
                last_before = messageData[-2]
                file_data = last_before.get("fileData", {})
                if file_data:
                    translated_paragraphs = [translate_text(p, 'en') for p in file_data.get("paragraph", [])]
                    translated_bullets = [translate_text(b, 'en') for b in file_data.get("bullets", [])]
                    translated_header = [translate_text(h, 'en') for h in file_data.get("header", [])]
                    translated_title = [translate_text(t, 'en') for t in file_data.get("title", [])]
                    translated_tables = [translate_text(tb, 'en') for tb in file_data.get("tables", [])]
                    content = {
                        "mainHeader": translated_header,
                        "title": translated_title,
                        "paragraph": translated_paragraphs,
                        "bullets": translated_bullets,
                        "table": translated_tables,
                        "message": "Here is the translated version of the document."
                    }
                    messageData.append(content)
                    request.session['messageData'] = messageData
                    return render(request, 'myFileReader/index.html', {'messageData': messageData})
            request.session['messageData'] = messageData
            return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # ---- 12) Disease helper ----
    if match_disease:
        from .disease import disease_model
        start_len = len(messageData)
        if Non_English and detected_language != "en":
            from .utils import translate_text

            def _tx(s: str) -> str:
                try:
                    tr = translate_text(s,
                                        target_language=detected_language)  # detected_language is a CODE like 'fr'
                    return getattr(tr, "text", None) or s
                except Exception:
                    return s

            # Translate strings in the items added by draw_square
            for i in range(start_len, len(messageData)):
                item = messageData[i]
                if isinstance(item, dict):
                    # translate top-level string fields
                    for k, v in list(item.items()):
                        if isinstance(v, str):
                            item[k] = _tx(v)
                        elif isinstance(v, dict):
                            # translate strings one level deep (e.g., solution subfields)
                            item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                        elif isinstance(v, list):
                            item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                elif isinstance(item, str):
                    messageData[i] = _tx(item)


        updated_messageData = disease_model(data_input, messageData)
        request.session["messageData"] = updated_messageData
        return render(request, 'myFileReader/index.html', {'messageData': updated_messageData})

    # ---- 13) Bar chart ----
    if Bar_match:
        from .utils import generate_barChart
        messageData = generate_barChart(data_input, messageData, request.session.get('messageData', []))
        start_len = len(messageData)
        if Non_English and detected_language != "en":
            from .utils import translate_text

            def _tx(s: str) -> str:
                try:
                    tr = translate_text(s,
                                        target_language=detected_language)  # detected_language is a CODE like 'fr'
                    return getattr(tr, "text", None) or s
                except Exception:
                    return s

            # Translate strings in the items added by draw_square
            for i in range(start_len, len(messageData)):
                item = messageData[i]
                if isinstance(item, dict):
                    # translate top-level string fields
                    for k, v in list(item.items()):
                        if isinstance(v, str):
                            item[k] = _tx(v)
                        elif isinstance(v, dict):
                            # translate strings one level deep (e.g., solution subfields)
                            item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                        elif isinstance(v, list):
                            item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                elif isinstance(item, str):
                    messageData[i] = _tx(item)


        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # ---- 14) Linear / Quadratic equations ----
    if match_linear:
        from .utils import linear_equation
        messageData = linear_equation(data_input, match_linear, messageData, request.session.get('messageData', []))
        start_len = len(messageData)
        if Non_English and detected_language != "en":
            from .utils import translate_text

            def _tx(s: str) -> str:
                try:
                    tr = translate_text(s,
                                        target_language=detected_language)  # detected_language is a CODE like 'fr'
                    return getattr(tr, "text", None) or s
                except Exception:
                    return s

            # Translate strings in the items added by draw_square
            for i in range(start_len, len(messageData)):
                item = messageData[i]
                if isinstance(item, dict):
                    # translate top-level string fields
                    for k, v in list(item.items()):
                        if isinstance(v, str):
                            item[k] = _tx(v)
                        elif isinstance(v, dict):
                            # translate strings one level deep (e.g., solution subfields)
                            item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                        elif isinstance(v, list):
                            item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                elif isinstance(item, str):
                    messageData[i] = _tx(item)


        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    if match_quadratic:
        from .utils import quadratic
        messageData = quadratic(data_input, match_quadratic, messageData, request.session.get('messageData', []))
        start_len = len(messageData)
        if Non_English and detected_language != "en":
            from .utils import translate_text

            def _tx(s: str) -> str:
                try:
                    tr = translate_text(s,
                                        target_language=detected_language)  # detected_language is a CODE like 'fr'
                    return getattr(tr, "text", None) or s
                except Exception:
                    return s

            # Translate strings in the items added by draw_square
            for i in range(start_len, len(messageData)):
                item = messageData[i]
                if isinstance(item, dict):
                    # translate top-level string fields
                    for k, v in list(item.items()):
                        if isinstance(v, str):
                            item[k] = _tx(v)
                        elif isinstance(v, dict):
                            # translate strings one level deep (e.g., solution subfields)
                            item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                        elif isinstance(v, list):
                            item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                elif isinstance(item, str):
                    messageData[i] = _tx(item)



        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    if match_quadratic_two:
        from .utils import quadratic_two
        messageData = quadratic_two(data_input, match_quadratic_two, messageData, request.session.get('messageData', []))
        start_len = len(messageData)
        if Non_English and detected_language != "en":
            from .utils import translate_text

            def _tx(s: str) -> str:
                try:
                    tr = translate_text(s,
                                        target_language=detected_language)  # detected_language is a CODE like 'fr'
                    return getattr(tr, "text", None) or s
                except Exception:
                    return s

            # Translate strings in the items added by draw_square
            for i in range(start_len, len(messageData)):
                item = messageData[i]
                if isinstance(item, dict):
                    # translate top-level string fields
                    for k, v in list(item.items()):
                        if isinstance(v, str):
                            item[k] = _tx(v)
                        elif isinstance(v, dict):
                            # translate strings one level deep (e.g., solution subfields)
                            item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                        elif isinstance(v, list):
                            item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                elif isinstance(item, str):
                    messageData[i] = _tx(item)

        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})


    # ---- 15) Summarise ----
    if match_summarise:
        from .utils import summarise_message
        messageData = summarise_message(data_input, messageData)
        request.session['messageData'] = messageData
        start_len = len(messageData)
        if Non_English and detected_language != "en":
            from .utils import translate_text

            def _tx(s: str) -> str:
                try:
                    tr = translate_text(s,
                                        target_language=detected_language)  # detected_language is a CODE like 'fr'
                    return getattr(tr, "text", None) or s
                except Exception:
                    return s

            # Translate strings in the items added by draw_square
            for i in range(start_len, len(messageData)):
                item = messageData[i]
                if isinstance(item, dict):
                    # translate top-level string fields
                    for k, v in list(item.items()):
                        if isinstance(v, str):
                            item[k] = _tx(v)
                        elif isinstance(v, dict):
                            # translate strings one level deep (e.g., solution subfields)
                            item[k] = {kk: _tx(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
                        elif isinstance(v, list):
                            item[k] = [_tx(x) if isinstance(x, str) else x for x in v]
                elif isinstance(item, str):
                    messageData[i] = _tx(item)

        request.session['messageData'] = messageData


        return render(request, 'myFileReader/index.html', {'messageData': messageData})
    try:
        # Ensure list
        messageData = request.session.get('messageData', [])
        if not isinstance(messageData, list):
            messageData = [messageData] if isinstance(messageData, dict) else []

        from .chat import pairs
        patterns = [pattern for pattern, _ in pairs]

        extracted = process.extractOne(data_input, patterns, scorer=fuzz.partial_ratio)
        if extracted:
            best_match, score, _ = extracted
            print("bess", best_match)
            print(score)

            if score > 78:
                # Find mapped response (English)
                resp_en = None
                for pattern, response in pairs:
                    if pattern == best_match:
                        resp_en = response[0] if isinstance(response, (list, tuple)) and response else str(response)
                        break

                if resp_en is None:
                    messageData.append({"message": "Matched, but no response found."})
                else:
                    # Translate back to user's language if original wasn't English
                    resp_out = resp_en
                    if Non_English and detected_language and detected_language.lower() != "en":
                        try:
                            from .utils import translate_text
                            tr = translate_text(resp_en, target_language=detected_language.lower())
                            resp_out = getattr(tr, "text", None) or (tr if isinstance(tr, str) else resp_en)
                        except Exception as e:
                            print("Back-translation failed:", e)

                    # Append exactly one message
                    messageData.append({"message": resp_out})

                    # ---- Optional tool routing based on EN user text (data_input) ----
                    weather_pattern = r"""(?ix)
                        \b(
                            (what('|s| is)?\s*(the\s*)?(weather|forecast|weather\s+report)) |
                            (can|could|will|would|do|does)\s+you\s+(tell|give)\s+me\s+(the\s*)?(weather|forecast) |
                            (is|was|will|would|does|do|did|are|am|were|being|be|has|have|had)\s+(it|the\s+weather)\s+(like|raining|snowing|sunny|cloudy|hot|cold|windy|nice) |
                            (rain|snow|sun|sunny|clouds|cloudy|storm|hot|cold|temperature|windy|humidity)
                        )
                        (\s+(today|tomorrow|tonight|this\s+(morning|afternoon|evening|weekend)|on\s+\w+day)?)?
                        (\s+(in|for)\s+[a-zA-Z\s]+)?
                        \??
                    """

                    try:
                        if re.search(weather_pattern, data_input):
                            from .api import get_weather, get_location
                            city = get_location()
                            if city:
                                weather = get_weather(city)
                                messageData.append({"weather": weather})

                        elif re.search(r"(triangle).*(area)|(area).*(triangle)", data_input, re.I):
                            from .drawShapes import draw_triangle
                            messageData = draw_triangle(messageData, data_input)

                        elif re.search(r"(rectangle).*(area)|(area).*(rectangle)", data_input, re.I):
                            from .drawShapes import draw_rectangle
                            messageData = draw_rectangle(messageData, data_input)

                    except Exception as e:
                        messageData.append({"message": f"Tool routing failed: {e}"})

                request.session['messageData'] = messageData
                return render(request, 'myFileReader/index.html', {'messageData': messageData})

        # No extraction / low score path falls through
    except Exception as e:
        messageData.append({'message': str(e)})
        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    # ---- 16) Pairs fuzzy matching + optional tool routing ----


    # ---- 17) Google Search fallback ----
    try:
        from .utils import google_search, translate_text
        results = google_search(data_input)

        if isinstance(results, list):
            # helper: deep-translate any str/list/dict into detected_language
            def _tx_value(v):
                if isinstance(v, str):
                    try:
                        tr = translate_text(v, target_language=detected_language)
                        return getattr(tr, "text", None) or v
                    except Exception:
                        return v
                if isinstance(v, list):
                    return [_tx_value(x) for x in v]
                if isinstance(v, dict):
                    return {k: _tx_value(val) for k, val in v.items()}
                return v

            for idx, res in enumerate(results):
                title = res.get('title', 'No Title')
                snippet = res.get('snippet', 'No snippet available')
                link = res.get('link', '')
                logo = res.get("pagemap", {}).get("cse_image", [{}])[0].get("src", "")

                # Skip bad/empty links
                if not link or not link.startswith(("http://", "https://")):
                    continue

                print(f"{idx + 1}. {title}\n📌 {snippet}\n🔗 {link}\n🖼️ Logo: {logo}\n")

                # try to fetch extra text from the page (optional)
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(link, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")
                        headlines = [h1.get_text(strip=True) for h1 in soup.find_all("h1")][:3]
                        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")][:5]
                    else:
                        headlines, paragraphs = [], []
                except Exception as e:
                    print(f"Error fetching content from {link}: {e}")
                    headlines, paragraphs = [], []

                mainHeader = headlines[0] if headlines else "No headline found"
                paragraph = [f"{i}. {p}" for i, p in enumerate(paragraphs, start=1)] if paragraphs else [
                    "No content found"]

                content = {
                    "mainHeader": mainHeader,
                    "search": data_input,  # this is already English per your earlier translation
                    "paragraph": paragraph,
                    "title": title,
                    "snippet": snippet,
                    "link": link,
                    "logo": logo,
                }

                # If original input wasn't English, translate *this content* to the user's language
                if Non_English and detected_language != "en":
                    content = _tx_value(content)

                # NOW append the website card
                messageData.append({"website": content})

                # If you only want the first result, uncomment:
                # break

        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

    except Exception as e:
        messageData.append({"message": f"Search failed: {e}"})
        request.session['messageData'] = messageData
        return render(request, 'myFileReader/index.html', {'messageData': messageData})


def clear_message_data(request):
    storageData=[]
    storageData.append(messageData)
    print("storageData",storageData)
    # Clear only the 'messageData' key from the session
    request.session.pop('messageData', None)
    return JsonResponse({"status": "cleared"})







def upload_datafiles(request):
    from .utils import upload_files

    if request.method == "POST":
        # Reset messageData in the session
        request.session["messageData"] = []  # Directly reset to an empty list

        uploaded_file = request.FILES.get("file")

        # Initialize messageData (it will be empty if not set before)
        messageData = request.session.get("messageData", [])

        # Call the file processing utility function
        messageData = upload_files(uploaded_file, messageData, request)

        # Handle any errors or issues with file processing (you can add your error handling here)

        # Save updated messageData to session
        request.session["messageData"] = messageData

        # Render the updated message data to the template
        return render(request, 'myFileReader/index.html', {'messageData': messageData})

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Automatically log in the user after signup
            return redirect("home")
    else:
        form = RegisterForm()
    return render(request, "register.html", {"form": form})


# User Login View
def user_login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                next_url = request.GET.get("next", "home")  # Redirect to next page if available
                return redirect(next_url)
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


# User Logout View
def user_logout(request):
    logout(request)
    return redirect("login")  # Redirect to login page after logout


# File Upload & Text Submit View (Merged)

