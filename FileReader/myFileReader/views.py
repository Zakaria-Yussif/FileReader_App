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
    """Processes user input and generates a bar chart"""
    global messageData
    global messageInfo
    global complete
    if request.method == "POST":
        data_inputData = request.POST.get('message', '').strip()
        # file_input=request.FILES['file']
        from .utils import  detect_language


        data_input = re.sub(r"\s+", " ", data_inputData).strip()

        print("daa",data_input)
        actions = ["draw", "generate","show","solve", "calculate", "compute", "if", "determine", "evaluate", "work out", "find"]
        shapes = ["triangle", "square", "circle", "rectangle", "star"]

        action_pattern = r"|".join([re.escape(a) for a in actions])
        shape_pattern = r"|".join([f"{s}s?" for s in shapes])
        draw_shape_regex = rf"\b(?:{action_pattern})\b.*?\b({shape_pattern})\b"  # group the shape

        match_draw_shapes = re.search(draw_shape_regex, data_input, re.IGNORECASE)

        easy_pattern = r"\b(write|generate|create|need).*?\bessay\b"
        translate_pattern = r"(?:please\s+)?(?:can you\s+)?translate\s+(.+?)\s+(?:to|in|into)\s+(\w+)"

        pattern = r"""^Solve for x: \d+x \+ \d+ = \d+,x = -?\d+$|
                                  ^John has \d+ apples and buys \d+ more\. How many apples does he have\?,-?\d+$|
                                  ^Find the derivative of \d+x\^\d+,-?\d+x\^\d+$|
                                  ^Evaluate f\(x\) = x\^2 - \d+x \+ \d+ at x = -?\d+,f\(-?\d+\) = -?\d+$"""


          # Output: True

        confirmation_pattern = r"\b(yes|yeah|yep|sure|ok|okay|affirmative|absolutely|certainly|add\s+image\s+\d+)\b"

        Bar_pattern = r"^\s*\w+\s*:\s*\d+\s*(,\s*\w+\s*:\s*\d+\s*)*$"
        linear_pattern = r"^([-+]?\d*\.?\d+)?\s*\*?\s*x\s*([-+]?\d*\.?\d+)?\s*\*?\s*y?\s*([-+]?\d+(\.\d+)?)?\s*=\s*([-+]?\d*\.?\d+)$"
        # linear_pattern2 = linear_pattern = r"^y\s*=\s*([-+]?\d*\.?\d+)?\s*\*?\s*x\s*([-+]?\s*\d+(\.\d+)?)?$"
        diseases_pattern = re.compile(
            r"""
            ^                              # start of string
            (?:.*?\b(?:have|got|experiencing|suffering\s+from)\b)  # any lead‑in
            \s*
            (?P<symptoms>                  # capture all following text as “symptoms”
                (?:                        # non‑capturing group for repetition
                    [^,;/\|\n\r]+?         # any chars except separators
                )
                (?:                        # then zero or more of…
                    (?:[,;/\|\n\r]|       #   any one of: comma, semicolon, slash, pipe, newline
                     \band\b|              #   or the word “and”
                     \bor\b|               #   or the word “or”
                     \band/or\b)           #   or “and/or”
                    [^,;/\|\n\r]+?         #   followed by more symptom text
                )*
            )
            \s*$                           # optional trailing space, end
            """,
            re.IGNORECASE | re.VERBOSE
        )
        summarise=r"\b(?:please\s*)?(?:can|could|would)?\s*(you\s*)?(please\s*)?(summarize|summarise)\b"
        # solve_pattern = re.compile(r'^[0-9a-zA-Z\+\-\*/\^\=\.\s\(\)]+$')

        quadratic_pattern = r"^([-+]?\d*\.?\d+|\d+)x\^2\s*([-+]?\d*\.?\d+|\d*)x\s*([-+]?\d*\.?\d+|\d+)?$"
        quadratic_pattern_two = r"^([-+]?\d*\.?\d+|\d+)\s*x\^2\s*([-+]?\d*\.?\d+|\d+)?\s*x\s*([-+]?\d*\.?\d+|\d+)?\s*(?:[:;,]\s*|\s*and\s*)\s*([-+]?\d*\.?\d+|\d+)\s*x\^2\s*([-+]?\d*\.?\d+|\d+)?\s*x\s*([-+]?\d*\.?\d+|\d+)?$"
        non_english_pattern = r"[^\x00-\x7F]"
        draw_images_patternData = r"(?:please\s*)?(?:can|could|would)?\s*(?:you\s*)?(?:please\s*)?(draw|show|generate|create)\s+(?:an?\s*)?(?:image|picture|photo|drawing)?\s*(?:of|about)?\s*(?P<description>.+)"

        match_summarise=re.search(summarise, data_input, re.IGNORECASE)
        # match_drawShapes=re.search(draw_pattern,data_input, re.IGNORECASE)
        matches_none_English = re.findall(non_english_pattern, data_input)
        match_essay = re.search(easy_pattern, data_input, re.IGNORECASE)
        Bar_match = re.search(Bar_pattern, data_input, re.IGNORECASE)
        match_translate_Data = re.search(translate_pattern, data_input, re.IGNORECASE)
        match_linear = re.search(linear_pattern, data_input, re.IGNORECASE)
        match_quadratic = re.search(quadratic_pattern, data_input, re.IGNORECASE)
        match_quadratic_two = re.search(quadratic_pattern_two, data_input, re.IGNORECASE)
        match_draw_imageData=re.search(draw_images_patternData, data_input.strip(), re.IGNORECASE)
        match_disease=re.findall(diseases_pattern,data_input)
        match_solve_partern=re.search(pattern, data_input, re.IGNORECASE)

        match_translate= re.search(confirmation_pattern, data_input, re.IGNORECASE)

        messageData = request.session.get('messageData', [])
        # Process text with SpaCy
        data_input = data_input.lower()

        # Clean input
        data_input = re.sub(r"\s+", " ", data_input).strip()

        if not data_input:
            if isinstance(messageData, list):
                messageData.append({"message": "⚠️ Please enter a valid sentence!"})
            else:
                messageData = [{"message": "⚠️ Please enter a valid sentence!"}]
            return render(request, 'myFileReader/index.html', {'messageData': messageData})

        message = deepcopy(data_input)
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
            for i, token in enumerate(doc):
                if token.pos_ == "NOUN" or token.dep_ == "nsubj":
                    missing_word_position = i + 1
                    tokens.insert(i + 1, "[MASK]")
                    break
        elif not has_object:
            for i, token in enumerate(doc):
                if token.pos_ == "VERB":
                    missing_word_position = i + 1
                    tokens.insert(i + 1, "[MASK]")
                    break

        if missing_word_position is None:
            tokens.append("[MASK]")
            missing_word_position = len(tokens) - 1

        masked_sentence = " ".join(tokens)
        masked_sentence = masked_sentence.replace("[MASK]", "", masked_sentence.count("[MASK]") - 1)

        # Error if more than one [MASK] token
        if masked_sentence.count("[MASK]") != 1:
            messageData.append({"message": "⚠️ Error: More than one [MASK] token found."})
            return render(request, 'myFileApp/index.html', {'messageData': messageData})

        # 🔹 Predict missing word

        prediction = fill_mask(masked_sentence)
        predicted_word = prediction[0]["token_str"].strip()
        tokens[missing_word_position] = predicted_word
        complete_sentence = " ".join(tokens).strip()

        if isinstance(messageData, list):
            messageData.append({"message": message, "id": 123})
            request.session['messageData'] = messageData
            detect_language = detect_language(data_input)




        # if matches_none_English:
        #     messageData.append({"message":complete_sentence})
        #     from .utils import translate_text
        #     full_language_name = langcodes.Language.get(detect_language).language_name()
        #     translator =translate_text(data_input,target_language="en")
        #
        #     translator.text= translator.text
        #     translator.src=translator.src
        #     translator.dest=translator.dest
        #     trans_message = {
        #        "message": f"This is written in {full_language_name}",
        #        "user_input": data_input,
        #         "translator": f"This is the translated sentence in English:",
        #         "translator_text": translator.text
        #     }

            # messageData.append({"translated": trans_message })
            # request.session['messageData'] = messageData
            # return render(request, 'myFileReader/index.html', {'messageData': messageData})
        if confirmation_pattern:
            messageData=request.session['messageData']
            last_Data= messageData[-2] if len(messageData) >= 2 else "none"
            print("last_Data", last_Data)


        if match_solve_partern:
            from .mathModel import predict
            data_inputData=predict(question=data_input)
            print("data_input",data_input)
            content={
                  "equation": data_input,
                  "step3": data_inputData,
                  "answer": data_inputData,
                }
            messageData.append({"solution":content})


        if match_essay:
            messageData.append({"message": complete_sentence})
            from .utils import generate_essay
            essayData = generate_essay(data_input)
            messageData.append({"message": essayData})
            # request.session['messageData'] = messageData



            return render(request, 'myFileReader/index.html', {'messageData': messageData})
        # elif math_model:
        #    messageData.append({"message": complete_sentence})
        #      from .mathModel import solve_math
        #      data_inputData=solve_math(data_input)
        #      print("data_input",data_input)
        #      content={
        #          "equation": data_input,
        #          "step3": data_inputData,
        #          "answer": data_inputData,
        #      }
        #      messageData.append({"solution":content})
        elif match_draw_imageData:
            from .draw_images import google_search_image
            description=match_draw_imageData.group("description").strip()

            print("dataDected",description)
            data_input=description


            query=description
            results= google_search_image(query)
            # print("dataDected", data_input)
            print("results", results)
            if isinstance(results, list):
                for index, result in enumerate(results):
                    link = result.get("link") if isinstance(result, dict) else "result"
                    if not link:
                        link = "None"
                    label = f"image{index + 1}:link"
                    print(label, link)
                    messageData.append({
                        "image_url": link,
                        "label": label
                    })

                # Append the message only once after processing all items
                message = "Which of the following images  would you like me to draw for you?"
                messageData.append({"last_Data": message})




        elif match_draw_shapes:
            messageData = request.session.get('messageData', [])
            shapes = match_draw_shapes.group(1).lower().rstrip('s')





            if  shapes=="triangle":
                messageData.append({"message": complete_sentence})
                from .drawShapes import draw_triangle
                messageData = draw_triangle(messageData,data_input)

            elif  shapes=="square" :
                messageData.append({"message": complete_sentence})
                from .drawShapes import draw_square
                messageData = draw_square(messageData,data_input)
                # request.session['messageData'] = messageData
                return render(request, 'myFileReader/index.html', {'messageData': messageData})
            elif shapes=="rectangle" :
                messageData.append({"message": complete_sentence})
                from .drawShapes import draw_rectangle
                messageData = draw_rectangle(messageData ,data_input)
                # request.session['messageData'] = messageData
                return render(request, 'myFileReader/index.html', {'messageData': messageData})
            elif shapes=="circle" :
                from .drawShapes import draw_circle
                messageData = draw_circle(messageData ,data_input)
                # request.session['messageData'] = messageData
                return render(request, 'myFileReader/index.html', {'messageData': messageData})








        elif match_translate_Data:
            messageData = request.session.get('messageData', [])
            from .utils import translate_text

            cleaned_sentence = re.sub(
                r'\b(translate|can you|can|into|please|in)\b',
                '',
                data_input,
                flags=re.IGNORECASE
            ).strip()
            print("cleaned_sentence", cleaned_sentence)
            cleaned_sentence_lower = cleaned_sentence.lower().strip()

            # Split into words to get the last one (or last two)
            words = cleaned_sentence_lower.split()
            last_word = words[-1] if words else ''
            last_two_words = " ".join(words[-2:]) if len(words) >= 2 else last_word

            detected_language = None
            phrase = cleaned_sentence_lower

            print("last_word", phrase)

            for code, name in LANGUAGES.items():
                name_lower = name.lower()

                # Match by name or code
                if last_word == name_lower or last_two_words == name_lower or last_word == code.lower():
                    detected_language = code
                    # Remove the language from the phrase
                    if name_lower in cleaned_sentence_lower:
                        phrase = cleaned_sentence_lower.replace(name_lower, '').strip()
                    elif code.lower() in cleaned_sentence_lower:
                        phrase = cleaned_sentence_lower.replace(code.lower(), '').strip()

                    break


            if detected_language:

                translator= translate_text(phrase,target_language=detected_language)
                translator.text = translator.text
                translator.src = translator.src
                translator.dest = translator.dest
                trans_message = {
                    # "message": f"This is written in {LANGUAGES[detected_language]}",
                    "user_input": data_input,
                    "translator": f"This is the translated sentence into {LANGUAGES[detected_language]}:",
                    "translator_text": translator.text
                }


                messageData.append({"translated": trans_message})
                request.session['messageData'] = messageData

                print("✅ Detected language:", LANGUAGES[detected_language])
                print("📝 Language code:", detected_language)
                print("🗣️ Phrase to translate:", phrase)
            else:
                print("❌ Could not detect language.")
                message="This language is not supported yet. Please try again with a different language."
                messageData.append({"message": message})
                if len(messageData) >= 2:
                    last_before = messageData[-2]
                    file_data = last_before.get("fileData", {})
                    print("last_before", last_before)
                    print("✅ This is the data before the translated version:")
                    print("Main Header:", file_data.get("mainHeader"))
                    print("paragraphs:", file_data.get("paragraph"))
                    print("Message:", file_data.get("message"))
                        #
                    translated_paragraphs = [translate_text(p, 'en') for p in file_data.get("paragraph", [])]
                    translated_bullets = [translate_text(b, 'en') for b in file_data.get("bullets", [])]
                    translated_header =[translate_text(h, 'en') for h in file_data.get("header", [])]
                    translated_title=[translate_text(t, 'en') for t in file_data.get("title", [])]
                    translated_tables=[translate_text(tb,'en') for tb in file_data.get("tables", []) ]
                    content = {
                         "mainHeader": translated_header,
                         "title": translated_title,
                         "paragraph": translated_paragraphs,
                        "bullets": translated_bullets,
                        "table": translated_tables,
                        "message": "Here is the translated version of the document."
                     }
                    print({"data is file":content})

                    messageData.append(content)
                    # request.session['messageData'] = messageData
                    return render(request, 'myFileReader/index.html', {'messageData': messageData})
                else:

                   print("Not enough entries in messageData.")

        elif match_disease:
            messageData = request.session.get('messageData', [])

            # Import your model or function that handles disease prediction
            from .disease import disease_model  # assuming disease_model is a function

            # Use the model to update messageData
            updated_messageData = disease_model(data_input, messageData)

            # Save updated data in session
            request.session["messageData"] = updated_messageData

            # Render with updated context
            return render(request, 'myFileReader/index.html', {'messageData': updated_messageData})

        elif Bar_match:
            from .utils import generate_barChart
            messageData = request.session.get('messageData', [])
            messageData=generate_barChart(data_input,messageData, request.session['messageData'])
            return render(request, 'myFileReader/index.html', {'messageData': messageData})

                # LINEAR EQUATIO
        elif match_linear:
            from .utils import linear_equation


            print("data_input",data_input)
            messageData = request.session.get('messageData', [])
            messageData=linear_equation(data_input,match_linear, messageData, request.session['messageData'])
            return render(request, 'myFileReader/index.html', {'messageData': messageData})

        elif match_quadratic:
            from .utils import quadratic
            messageData = request.session.get('messageData', [])
            messageData=quadratic(data_input,match_quadratic,messageData, request.session['messageData'])
            return render(request, 'myFileReader/index.html', {'messageData': messageData})

        elif match_quadratic_two:
            from .utils import quadratic_two
            messageData = request.session.get('messageData', [])
            messageData=quadratic_two(data_input,match_quadratic_two,messageData, request.session['messageData'])
            return render(request, 'myFileReader/index.html', {'messageData': messageData})
        elif match_summarise:
            from .utils import summarise_message
            messageData = request.session.get('messageData', [])
            messageData=summarise_message(data_input,messageData)
        else:
            from .utils import google_search
            messageData = request.session.get('messageData', [])

            findOne = "(how do|what's the method for) calculating the area of a triangle\??"
            findTwo = "how can I (find|calculate) the area of a triangle\??"
            findRectangle = "(how do|what's the method for) calculating the area of a rectangle\??"
            findRectangleTwo = " how can I (find|calculate) the area of a rectangle\??"



            # Check for subject, verb, and object

            # If messageInfo was set (i.e., prediction succeeded)
            if messageInfo:
                messageData.append({"message": messageInfo})
                # request.session["messageData"] = messageData
                return render(request, 'myFileReader/index.html', {'messageData': messageData})

            # If prediction did not occur (fallback to pattern matching)
            from .chat import pairs
            if matches_none_English:
                from .utils import translate_text
                detected_language = None


                for code, name in LANGUAGES.items():
                    name_lower = name.lower()
                    detected_language=code
                    print("detected_language",detected_language)

                full_language_name = langcodes.Language.get(detect_language).language_name()
                print(full_language_name)
                translator = translate_text(data_input, target_language="en")

                translator.text = translator.text
                translator.src = translator.src
                translator.dest = translator.dest
                print("text", translator.text)


                messageData = request.session.get('messageData', [])
                pairs= pairs

                patterns = [pattern for pattern, _ in pairs]

                best_match, score, _ = process.extractOne(translator.text, patterns, scorer=fuzz.partial_ratio)
                print("bess",best_match)
                print(score)


            # If a pattern match is found
                if score > 75:
                  for pattern, response in pairs:
                    if pattern == best_match:

                        translator = translate_text(response[0], target_language=full_language_name)
                        translator.text = translator.text
                        translator.src = translator.src
                        print("textDD", translator.text)
                        messageData.append({"message": translator.text})

                        request.session['messageData'] = messageData

                        weather_pattern = r"""(?ix)                 # ignore case, allow verbose mode
                            \b(
                                (what('|s| is)?\s*(the\s*)?(weather|forecast|weather\s+report)) |
                                (can|could|will|would|do|does)\s+you\s+(tell|give)\s+me\s+(the\s*)?(weather|forecast) |
                                (is|was|will|would|does|do|did|are|am|were|being|be|has|have|had)\s+(it|the\s+weather)\s+(like|raining|snowing|sunny|cloudy|hot|cold|windy|nice) |
                                (rain|snow|sun|sunny|clouds|cloudy|storm|hot|cold|temperature|windy|humidity)
                            )
                            (
                                \s+(today|tomorrow|tonight|this\s+(morning|afternoon|evening|weekend)|on\s+\w+day)?
                            )?
                            (
                                \s+(in|for)\s+[a-zA-Z\s]+     # simple location capture
                            )?
                            \??
                        """
                        findOne = "(how do|what's the method for) calculating the area of a triangle\??"
                        findTwo = "how can I (find|calculate) the area of a triangle\??"
                        findRectangle = "(how do|what's the method for) calculating the area of a rectangle\??"
                        findRectangleTwo = " how can I (find|calculate) the area of a rectangle\??"





                        search_weather_pattern= re.search(weather_pattern, best_match, re.IGNORECASE)


                        if search_weather_pattern  or "weather"  in response[0].lower():
                            from .api import get_weather, get_location
                            city = get_location()

                            if city:
                                weather = get_weather(city)
                                print("weather", weather)
                                messageData.append({"weather": weather})
                                # request.session['messageData'] = messageData


                        elif best_match == findOne or best_match == findTwo:
                            from .drawShapes import draw_triangle
                            messageData = draw_triangle(messageData,data_input)
                            # request.session['messageData'] = messageData

                        elif best_match == findRectangleTwo:
                             from .drawShapes import draw_rectangle
                             messageData = draw_rectangle(messageData,data_input)
                             messageData.append(messageData)
                             # request.session['messageData'] = messageData





                        return render(request, 'myFileReader/index.html', {'messageData': messageData})
            else:
                    messageData = request.session.get('messageData', [])
                    pairs = pairs

                    patterns = [pattern for pattern, _ in pairs]

                    best_match, score, _ = process.extractOne(data_input, patterns, scorer=fuzz.partial_ratio)
                    print("bess", best_match)
                    print(score)

                    # If a pattern match is found
                    if score > 75:
                        for pattern, response in pairs:
                            if pattern == best_match:

                                messageData.append({"message": response[0]})
                                request.session['messageData'] = messageData

                                weather_pattern = r"""(?ix)                 # ignore case, allow verbose mode
                                                \b(
                                                    (what('|s| is)?\s*(the\s*)?(weather|forecast|weather\s+report)) |
                                                    (can|could|will|would|do|does)\s+you\s+(tell|give)\s+me\s+(the\s*)?(weather|forecast) |
                                                    (is|was|will|would|does|do|did|are|am|were|being|be|has|have|had)\s+(it|the\s+weather)\s+(like|raining|snowing|sunny|cloudy|hot|cold|windy|nice) |
                                                    (rain|snow|sun|sunny|clouds|cloudy|storm|hot|cold|temperature|windy|humidity)
                                                )
                                                (
                                                    \s+(today|tomorrow|tonight|this\s+(morning|afternoon|evening|weekend)|on\s+\w+day)?
                                                )?
                                                (
                                                    \s+(in|for)\s+[a-zA-Z\s]+     # simple location capture
                                                )?
                                                \??
                                            """
                                findOne = "(how do|what's the method for) calculating the area of a triangle\??"
                                findTwo = "how can I (find|calculate) the area of a triangle\??"
                                findRectangle = "(how do|what's the method for) calculating the area of a rectangle\??"
                                findRectangleTwo = " how can I (find|calculate) the area of a rectangle\??"

                                search_weather_pattern = re.search(weather_pattern, best_match, re.IGNORECASE)

                                if search_weather_pattern or "weather" in response[0].lower():
                                    from .api import get_weather, get_location
                                    city = get_location()

                                    if city:
                                        weather = get_weather(city)
                                        print("weather", weather)
                                        messageData.append({"weather": weather})
                                        # request.session['messageData'] = messageData


                                elif best_match == findOne or best_match == findTwo:
                                    from .drawShapes import draw_triangle
                                    messageData = draw_triangle(messageData, data_input)
                                    # request.session['messageData'] = messageData

                                elif best_match == findRectangleTwo:
                                    from .drawShapes import draw_rectangle
                                    messageData = draw_rectangle(messageData, data_input)
                                    messageData.append(messageData)
                                    # request.session['messageData'] = messageData

                                return render(request, 'myFileReader/index.html', {'messageData': messageData})

        # 🔹 Predict missing word

            from .utils import google_search
            query = data_input
            results = google_search(query)
            messageData = request.session.get('messageData', [])


            if isinstance(results, list):
                for idx, result in enumerate(results):
                    title = result.get('title', 'No Title')
                    snippet = result.get('snippet', 'No snippet available')
                    link = result.get('link', 'No link available')
                    logo = result.get("pagemap", {}).get("cse_image", [{}])[0].get("src", "No logo available")

                    message = (f"{idx + 1}. {title}\n📌 {snippet}\n🔗 {link}\n🖼️ Logo: {logo}\n")
                    print(message)

                    try:
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        }
                        response = requests.get(link, headers=headers, timeout=5)

                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, "html.parser")
                            headlines = [h1.get_text(strip=True) for h1 in soup.find_all("h1")][:3]
                            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")][:5]

                            mainHeader = headlines[0] if headlines else "No headline found"
                            paragraph = [f"{i}. {p}" for i, p in enumerate(paragraphs, start=1)] if paragraphs else [
                                "No content found"]

                            content = {
                                "mainHeader": mainHeader,
                                "search": data_input,
                                "paragraph": paragraph,
                                "title": title,
                                "snippet": snippet,
                                "link": link,
                                "logo": logo,
                            }
                            print("content",content)
                            messageData.append({"website": content})
                            request.session["messageData"] = messageData

                    except Exception as e:
                        print(f"Error fetching content from {link}: {e}")

            # Store all results in session after processing






                # Set messageInfo if prediction is successful

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

