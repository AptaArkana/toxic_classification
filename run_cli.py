import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from test_cli import TextToxicityChecker

def main():
    # Dummy tokenizer, device, and model for illustration purposes
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./NAMEDIRECTORIMODEL')

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./NAMEDIRECTORIMODEL')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    # Get user input for text to check toxicity
    example_text = input("Enter the text to check toxicity: ")
    
    # Create an instance of TextToxicityChecker
    toxicity_checker = TextToxicityChecker(tokenizer, device, model)

   # Check toxicity using the class method
    result = toxicity_checker.check_text_toxicity(example_text)

    # Print the result
    print(result)

if __name__ == "__main__":
    main()
