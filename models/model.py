from logger import logging

import torch.nn as nn
import torch

logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./model')

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path='./model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def check_text_toxicity(text) -> dict:
    try:
        logger.info('Started checking toxicity...')
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors='pt').to(device)
        
        # Forward pass through the model
        outputs = model(**inputs)
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(outputs.logits)
        probabilities = probabilities.to('cpu').detach().numpy()
        id2label = model.config.id2label
        index = 0
        result = {
            'status': 1,
            'message': 'Request successful',
            'response': {
                'Konten_kasar': False,
                'Bukan_konten_kasar': False
            },
            'probabilities': {}
        }
        
        for _, v in id2label.items():
            # Check if probability exceeds the threshold (e.g., 0.85)
            if probabilities[0][index] > 0.85:
                result['response'][v] = True
                
            # Store the probability for each class
            result['probabilities'][v] = round(probabilities[0][index] * 100, 2)
            index += 1

        logger.info('Success...')
        
    except Exception as e:
        logger.error(f'Error: {e}')
        result = {
            'status': -1,
            'message': 'Error at the model level.',
            'response': {},
            'probabilities': {}
        }
    
    return result