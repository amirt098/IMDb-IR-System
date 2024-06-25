import torch
import json
import numpy as np


import wandb

from torch.utils.data import Dataset
from collections import Counter

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction

from transformers import default_data_collator
import evaluate

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
WAND_KEY = 'a965e082ae3823a8a3d8d30404eb4953471a51c9'

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        #  Implement initialization logic
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.encodings = None
        self.labels = None
        self.genre_to_label = {}
        self.label_to_genre = {}
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # Implement dataset loading logic
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

        if self.data is not None:
            logger.info("Loaded")
        else:
            logger.info("NOT Loaded")
            raise Exception("Dataset not loaded")

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # Implement genre filtering and visualization logic
        summaries = []
        genres = []

        for movie_id, movie_data in self.data.items():
            if 'first_page_summary' in movie_data and 'genres' in movie_data and movie_data['genres']:

                summary = movie_data['first_page_summary']
                genre = movie_data['genres'][0]
                if summary:
                    summaries.append(summary)
                    genres.append(genre)
                else:
                    logging.debug(f"null summary  for movie_id {movie_id}: {type(summary)}")

        logging.debug(f"# summaries collected: {len(summaries)}")
        logging.debug(f"# genres collected: {len(genres)}")

        genre_counts = Counter(genres)
        top_genres = [genre for genre, count in genre_counts.most_common(self.top_n_genres)]

        filtered_data = [(s, g) for s, g in zip(summaries, genres) if g in top_genres]
        summaries, genres = zip(*filtered_data)

        logging.debug(f"Summaries: {summaries[:5]}")
        logging.debug(f"Genres: {genres[:5]}")

        self.genre_to_label = {genre: idx for idx, genre in enumerate(top_genres)}
        self.label_to_genre = {idx: genre for genre, idx in self.genre_to_label.items()}
        self.labels = [self.genre_to_label[genre] for genre in genres]

        self.encodings = self.tokenizer(list(summaries), truncation=True, padding=True)

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # Implement dataset splitting logic

        summaries = list(self.encodings['input_ids'])
        labels = list(self.labels)

        # Debugging the lengths of summaries and labels
        logging.debug(f"Summaries length: {len(summaries)}")
        logging.debug(f"Labels length: {len(labels)}")

        # Ensure the lengths are consistent
        assert len(summaries) == len(labels), "Summaries and labels must have the same length"

        # Split the data into training and test sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            summaries, labels, test_size=test_size, random_state=42
        )

        # Further split the training data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=val_size, random_state=42
        )

        # Creating datasets
        self.train_dataset = self.create_dataset({'input_ids': train_texts}, train_labels)
        self.eval_dataset = self.create_dataset({'input_ids': val_texts}, val_labels)
        self.test_dataset = self.create_dataset({'input_ids': test_texts}, test_labels)

        logging.debug(f"train dataset size: {len(self.train_dataset)}")
        logging.debug(f"validation dataset size: {len(self.eval_dataset)}")
        logging.debug(f"test dataset size: {len(self.test_dataset)}")

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # Implement BERT fine-tuning logic
        wandb.login(key=WAND_KEY)

        wandb.login(key=WAND_KEY)
        num_labels = len(self.genre_to_label)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        #  Implement metric computation logic
        metric = evaluate.load("accuracy")
        logits, labels = pred

        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)

        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")

        precision_score = precision.compute(predictions=predictions, references=labels, average="weighted")
        recall_score = recall.compute(predictions=predictions, references=labels, average="weighted")
        f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")

        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision_score['precision'],
            'recall': recall_score['recall'],
            'f1': f1_score['f1']
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # Implement model evaluation logic
        trainer = Trainer(model=self.model)
        eval_results = trainer.evaluate(self.test_dataset)
        print(eval_results)
        # print(f"test Accuracy: {eval_results['eval_accuracy']}")
        # print(f"test Precision: {eval_results['eval_precision']}")
        # print(f"test Recall: {eval_results['eval_recall']}")
        # print(f"test F1 Score: {eval_results['eval_f1']}")

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        #  Implement model saving logic
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # Implement item retrieval logic

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # Implement length computation logic
        return len(self.labels)
