import numbers
import re
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sympy import symbols, Eq, solve, simplify, pretty

from myFileReader.views import messageData

# Load dataset
import pandas as pd
from datasets import Dataset

# Load your dataset
import re
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv(r"C:\Users\zakar\PycharmFileExtractor\FileApp\Maths.dataset/data/cv_asdiv-a/fold0/dev.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Split numbers
df['numbers'] = df['Numbers'].astype(str).apply(lambda x: x.split())

# Replace 'number0', 'number1' with actual numbers in the Body
def replace_numbers(body, numbers):
    for i, num in enumerate(numbers):
        body = body.replace(f'number{i}', num)
    return body

df['Body'] = df.apply(lambda row: replace_numbers(row['Body'], row['numbers']), axis=1)

# Combine Body and Question Statement
df['input_text'] = df['Body'] + " " + df['Ques_Statement']
df['target_text'] = df['Answer'].astype(str)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['input_text', 'target_text']])

# Tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize dataset
def tokenize(example):
    model_input = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=128)
    label = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=16)
    model_input["labels"] = label["input_ids"]
    return model_input

tokenized_dataset = dataset.map(tokenize)

# Training configuration
training_args = TrainingArguments(
    output_dir="./math_t5",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# ---------- Prediction Utilities ----------

def extract_numbers(text):
    return re.findall(r'\d+', text)

def replace_numbers_in_input(text):
    numbers = extract_numbers(text)
    count = [0]
    def replace_match(m):
        val = f'number{count[0]}'
        count[0] += 1
        return val
    template = re.sub(r'\d+', replace_match, text)
    return replace_numbers(template, numbers)

def predict(question):
    # Extract numbers
    numbers = extract_numbers(question)

    # Replace numbers in the question with number0, number1, ...
    count = [0]

    def replace_match(m):
        val = f'number{count[0]}'
        count[0] += 1
        return val

    # Replace all numbers in the question with placeholders
    template = re.sub(r'\d+', replace_match, question)

    # Replace placeholders with actual numbers in the input
    final_input = replace_numbers(template, numbers)

    # Add T5 prompt
    input_text = "solve: " + final_input

    # Tokenize input and predict output
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
    output = model.generate(input_ids, max_length=16)

    # Decode the output tensor to human-readable text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Handle edge cases in the output (e.g., unexpected outputs)
    output_text = output_text.strip()

    # If output is incomplete, add a fallback message
    if not output_text or "number" in output_text:
        output_text = "Could not generate a valid answer."

    print("Predicted answer:", output_text)

    return output_text

# ---------- Example ----------
question = "Ellen has 6 more balls than Marin. Marin has 9 balls. How many balls does Ellen have?"
print("Prediction:", predict(question))


# Split the Question Numbers column into list of numbers
# df['numbers'] = df['Question Numbers'].astype(str).apply(lambda x: x.split())
#
# # Replace placeholders like number0, number1 in Body
# def replace_numbers(body, numbers):
#     for i, num in enumerate(numbers):
#         body = body.replace(f'number{i}', num)
#     return body
#
# df['Body'] = df.apply(lambda row: replace_numbers(row['Body'], row['numbers']), axis=1)
#
# # Create final input and target columns
# df['input_text'] = df['Body'] + " " + df['Ques_Statement']
# df['target_text'] = df['Answer'].astype(str)
#
# # Convert to Hugging Face Dataset
# dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
# print(dataset[0])
#


# train_df, val_df = train_test_split(df, test_size=0.1)
# train_dataset = Dataset.from_pandas(train_df)
# val_dataset = Dataset.from_pandas(val_df)
#
# # Load tokenizer & model
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
#
# # Helper to clean equations
# def clean_equation(eq):
#     eq = eq.lower().strip()
#     eq = re.sub(r'([=+\-*/^()])', r' \1 ', eq)  # Space around math symbols
#     eq = re.sub(r'\s+', ' ', eq)  # Remove extra spaces
#     return eq
#
# # Preprocessing function
# def preprocess(examples):
#     questions = ["solve: " + clean_equation(q) for q in examples["question"]]
#     model_inputs = tokenizer(questions, padding="max_length", truncation=True, max_length=128)
#
#     labels = tokenizer(examples["answer"], padding="max_length", truncation=True, max_length=64)
#     labels["input_ids"] = [
#         [(token if token != tokenizer.pad_token_id else -100) for token in label]
#         for label in labels["input_ids"]
#     ]
#
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# # Tokenize datasets
# train_dataset = train_dataset.map(preprocess, batched=True)
# val_dataset = val_dataset.map(preprocess, batched=True)
#
# # Metric function
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     labels = [[(token if token != -100 else tokenizer.pad_token_id) for token in label] for label in labels]
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     decoded_preds = [pred.strip().lower() for pred in decoded_preds]
#     decoded_labels = [label.strip().lower() for label in decoded_labels]
#
#     acc = sum([p == l for p, l in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
#     return {"accuracy": acc}
#
# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./math_model",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     save_strategy="epoch",
#     num_train_epochs=3,
#     logging_dir="./logs",
#
# )
#
# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics
# )
#
# # Train & save
# trainer.train()
# model.save_pretrained("./math_model")
# tokenizer.save_pretrained("./math_model")
#
# # Load for inference
# model = T5ForConditionalGeneration.from_pretrained("./math_model")
# tokenizer = T5Tokenizer.from_pretrained("./math_model")
#
# # Solver function
# def solve_math(data_input):
#     if not data_input or not isinstance(data_input, str):
#         print("❌ Invalid or empty input:", data_input)
#         return "Invalid input"
#
#     try:
#         input_text = "solve: " + clean_equation(data_input.strip())
#         print("📥 Input text:", input_text)
#
#         inputs = tokenizer.encode(input_text, return_tensors="pt")
#         print("🧾 Tokenized input IDs:", inputs)
#
#         outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
#         print("📤 Generated output IDs:", outputs)
#
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("✅ Decoded result:", result)
#         return result
#     except Exception as e:
#         print("❌ Error:", e)
#         return "Error solving problem"
#
#
#
#
# # def solve_equation_steps( data_input,messageData):
# #     x = symbols('x')
# #
# #     # Parse left and right side
# #     left, right = data_input.replace(' ', '').split('=')
# #
# #     # Convert to SymPy equation
# #     equation = Eq(eval(left), eval(right))
# #
# #     # Solve
# #     solution = solve(equation, x)
# #
# #     # Build steps manually
# #
# #     equation = Eq(eval(left), eval(right))
# #     step1 = f"🔢 Step 1: Original equation\n   {pretty(equation)}\n"
# #     step2 += f"✂️  Step 2: Subtract {b} → {a}x = {c}\n"
# #     step3 += f"➗ Step 3: Divide by {a} → x =\\frac{{{b}}}{{{m}}}\n"
# #     step4 =f"x =\\frac{{{b}}}{{{m}}}\n"
# #     solution += f"✅ Final Answer: x = {solution[0]}"
# #
# #     content={
# #         "equation": equation,
# #         "step1": step1,
# #         "step2":step2,
# #         "step3":step3,
# #         "step4":step4,
# #         "answer":f"x = {solution[0]}"
# #
# #     }
# #     for key, value in content.items():
# #         print(f"{key}: {value}")
# #
# #     # If you're in Jupyter, you can render the LaTeX equation in step3
# #     if 'display' in globals():
# #         display(Math(step3))
# #
# #     messageData.append({"solution":content})
# #     return messageData
#
#
#
# # Example
#
# # Example
# equation = "5*x + 6 = 9"
# steps, answer = solve_equation_steps(equation)
# print(steps)
#
#

