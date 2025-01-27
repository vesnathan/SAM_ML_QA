import json
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification

os.environ['TRANSFORMERS_CACHE'] = '/tmp/'

logging.getLogger().setLevel(logging.DEBUG)

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, questions, document_text, answers, tokenizer, max_length):
        self.questions = questions
        self.document_text = document_text
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]

        combined_text = f"{question} {self.document_text} {answer}"

        encoding = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def inference(loader, model, device):
    model.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            prediction = torch.argmax(logits, dim=1).item()

    return prediction


def lambda_handler(event, context):
    logging.info(f"event['body'] {event['body']}")
    input_data = json.loads(event['body'])
    logging.info(f"input_data {input_data}")
    question = input_data['question']
    document_text = input_data['document_text']
    answer = input_data['answer']

    # Load the pre-trained tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='/tmp/')

    # Load the model
    model = RobertaForSequenceClassification.from_pretrained('./')

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare the input for inference
    input_data = {
        'question': question,
        'document_text': document_text,
        'answer': answer
    }

    # Create a dataset for the input data
    inference_dataset = CustomDataset([input_data['question']], input_data['document_text'],
                                      [input_data['answer']], tokenizer, max_length=512)
    inference_loader = DataLoader(inference_dataset, batch_size=1)

    # Call the inference function
    try:
        prediction = inference(inference_loader, model, device)

        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps(prediction)
        }


    except Exception as e:
        logging.error(e)
        response = {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps('Error occurred: ' + str(e))
        }



    return response