from BertFinetuner_mask import BERTFinetuner

# Instantiate the class
bert_finetuner = BERTFinetuner('documents.json', top_n_genres=5)

# Load the dataset
bert_finetuner.load_dataset()

# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution()

# Split the dataset
bert_finetuner.split_dataset()

# Fine-tune BERT model
bert_finetuner.fine_tune_bert()

"""
{'
 eval_loss': 1.3361676931381226,
'eval_accuracy': 0.4340681362725451,
'eval_precision': 0.5026897140194923,
'eval_recall': 0.4340681362725451,
'eval_f1': 0.3346449230339283,
'eval_runtime': 158.3178,
'eval_samples_per_second': 15.759,
'eval_steps_per_second': 0.985,
'epoch': 1.0
}
{
'eval_loss': 1.0868537425994873,
'eval_accuracy': 0.556312625250501,
'eval_precision': 0.5623027818640981, 
'eval_recall': 0.556312625250501, 
'eval_f1': 0.5566987299182602, 
'eval_runtime': 314.641, 
'eval_samples_per_second': 7.93, 
'eval_steps_per_second': 0.496, 
'epoch': 2.0
}
{'eval_loss': 1.3024715185165405, 
'eval_accuracy': 0.5290581162324649, 
'eval_precision': 0.5989382357140508, 
'eval_recall': 0.5290581162324649, 
'eval_f1': 0.4902333174587944, 
'eval_runtime': 158.704, 
'eval_samples_per_second': 15.721, 
'eval_steps_per_second': 0.983, 
'epoch': 3.0}
{'eval_loss': 1.3117491006851196, 
'eval_accuracy': 0.5535070140280561, 
'eval_precision': 0.579463137771552, 
'eval_recall': 0.5535070140280561, 
'eval_f1': 0.554481100102594, 
'eval_runtime': 312.531, 
'eval_samples_per_second': 7.983, 
'eval_steps_per_second': 0.499, 
'epoch': 4.0}
{'eval_loss': 1.5852760076522827, 
'eval_accuracy': 0.5727454909819639, 
'eval_precision': 0.5799105960880501, 
'eval_recall': 0.5727454909819639, 
'eval_f1': 0.5746087199761549, 
'eval_runtime': 356.9751, 
'eval_samples_per_second': 6.989, 
'eval_steps_per_second': 0.437, 
'epoch': 5.0}

{'train_runtime': 2991.5607, 
'train_samples_per_second': 4.17, 
'train_steps_per_second': 0.261, 
'train_loss': 0.865917489467523, 
'epoch': 5.0}
"""

# Compute metrics
bert_finetuner.evaluate_model()


# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')

