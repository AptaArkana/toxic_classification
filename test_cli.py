from logger import logging
import torch.nn as nn
import torch

class TextToxicityChecker:
    def __init__(self, tokenizer, device, model, threshold=0.85):
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def check_text_toxicity(self, text) -> dict:
        try:
            self.logger.info('Started checking toxicity...')
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            
            # Forward pass through the model
            outputs = self.model(**inputs)
            sigmoid = nn.Sigmoid()
            probabilities = sigmoid(outputs.logits)
            probabilities = probabilities.to('cpu').detach().numpy()
            id2label = self.model.config.id2label
            index = 0
            result = {
                'prediction': {
                    'label': None,
                    'probability': None
                }
            }
            
            for _, v in id2label.items():
                # Check if probability exceeds the threshold
                if probabilities[0][index] > self.threshold:
                    result['prediction']['label'] = 'Toxic' if v == 'Konten_kasar' else 'Polite'
                    result['prediction']['probability'] = round(probabilities[0][index] * 100, 2)
                    
                    # Since we have found a label with probability above threshold, break the loop
                    break
                
                index += 1

            self.logger.info('Success...')
            
        except Exception as e:
            self.logger.error(f'Error: {e}')
            result = {
                'message': 'Error at the model level.',
                'prediction': {
                    'label': None,
                    'probability': None
                }
            }
        
        return result