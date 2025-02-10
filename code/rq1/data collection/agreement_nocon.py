import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

class LogitsCalculator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.choices = ['Agree', 'Disagree']

    def calculate_logits(self, statement):
        # Prepare the prompt directly with the statement
        prompt = (
            "State your agreement with the statement.\n"
            f"Statement: {statement}\n"
            f"Choose from: {', '.join(self.choices)}\n"
            "Answer:"
        )

        # Encode input and get model output
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Get the logits for the last token
        last_token_logits = logits[0, -1, :]

        # Get token IDs for choices 'Agree' and 'Disagree'
        token_ids = [self.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in self.choices]
        logits_result = {choice: last_token_logits[token_id].item() for choice, token_id in zip(self.choices, token_ids)}

        return logits_result

    def process_statements(self, statements, output_file):
        # Check if the file exists to load previous results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}

        # Calculate the total number of iterations for tqdm progress bar
        total_iterations = len(statements)

        # Wrap the loop with tqdm for progress tracking
        with tqdm(total=total_iterations, desc="Processing statements") as pbar:
            # Loop through each statement
            for statement in statements:
                statement_text = statement[0]

                if statement_text not in all_results:
                    logits_result = self.calculate_logits(statement_text)

                    # Store results in the expected format
                    all_results[statement_text] = {
                        "logits": logits_result
                    }

                # Update the progress bar after each statement
                pbar.update(1)

            # Save results incrementally
            self.save_to_json(all_results, output_file)

    def save_to_json(self, data, output_file):
        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    model_names = [
        "llms/Meta-Llama-3-8B",
        'Meta-Llama-2-7B'
    ]

    for model_name in model_names:
        # Load statements data from a JSON file
        with open("data/temp_rev.json") as json_file:
            data = json.load(json_file)
        statements = data["AQ-27-all"]["statements"]

        calculator = LogitsCalculator(model_name)

        # Process statements
        output_file = f"./{model_name.replace('/', '_')}_NOCON_logits_1610_1.json"
        print(f"Processing model: {model_name}")
        calculator.process_statements(statements, output_file)
