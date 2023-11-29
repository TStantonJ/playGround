# =================================================================
# Class_Ex1:
# Check the class 3-LSM_Sentiment_Analysis.py code.
# Identify why the accuracy will not increase and why the network is not learning.
# Add pad pack sequence and use glove embedding and check the results again.
# This time you should see the validation accuracy increase and the loss goes down.


# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Download the swedish name dataset.
# classifying common swedish names into gender categories.
# Use Char level embedding.
# Then classify each name
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.optimizers import Adam
from sklearn.metrics import classification_report
df = pd.read_csv('Name_Data_set.csv', lines=True)

# PREP --------------------------------------------------------------------------------------------------------------
texts = df['namn'].values
labels = df['gender'].values

# load pretrained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Split first
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize
input_ids_train = []
input_ids_test = []
attention_masks_train = []
attention_masks_test = []

for text in texts_train:
    # Tokenize train
    encoded_dict = tokenizer.encode_plus(text, # input texts                     
                            add_special_tokens = True, # add [CLS] at the start, [SEP] at the end
                            max_length = 128, # if input text is longer, then it gets truncated
                            padding = 'max_length', # if input text is shorter, then it gets padded to 128
                            return_attention_mask = True,   
                            return_tensors = 'tf',
                            truncation=True)

    input_ids_train.append(encoded_dict['input_ids'][0]) 
    attention_masks_train.append(encoded_dict['attention_mask'][0])

for text in texts_test:
    # Tokenize test
    encoded_dict = tokenizer.encode_plus(text,                      
                            add_special_tokens = True, 
                            max_length = 128,           
                            padding = 'max_length',
                            return_attention_mask = True,   
                            return_tensors = 'tf',
                            truncation=True)  

    input_ids_test.append(encoded_dict['input_ids'][0])
    attention_masks_test.append(encoded_dict['attention_mask'][0])

# MODEL RUN --------------------------------------------------------------------------------------------------------------
# Model evaluation for scores
y_pred_logits = model.predict([input_ids_test, attention_masks_test]).logits
#y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
#y_pred_labels = tf.argmax(y_pred_logits, axis=1).numpy()

# Creating DataFrame
texts_test_series = pd.Series(texts_test, name='Text')
#scores_df = pd.DataFrame(y_pred_scores, columns=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
#final_df = pd.concat([texts_test_series, scores_df], axis=1)

# Adding overall score
#final_df['Overall_Score'] = final_df[['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']].max(axis=1)

# Save the model's architecture and weights; save the tokenizer
model.save('bert_emotion_classifier')
tokenizer.save_pretrained('bert_emotion_classifier_tokenizer')

# Load the saved tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert_emotion_classifier_tokenizer')
#model = TFBertForSequenceClassification.from_pretrained('bert_emotion_classifier')


# MODEL OUTPUT & EVAL --------------------------------------------------------------------------------------------------------------
# get classification report
#report = classification_report(labels_test, y_pred_labels, target_names=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'], output_dict=True)
#report_df = pd.DataFrame(report).transpose()

# report to csv
#report_df.to_csv('hug_data_sample_report.csv', index=False)

# preview report
#print(report_df)
print(20 * '-' + 'End Q2' + 20 * '-')
