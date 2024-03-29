# Sentence Similarity
The sentence_similarity.py contsains the code to capture sentence similarity score of sentence pairs. The pipeline prepares features (similarity measures) to train a machine learning model and then to predict the similarity score. Pearson correlation is also used to evaluate the feature importance and the model performance. Different stages of the machine learning pipeline for this project will be discussed below.

1. Data Preprocessing: 
At first text is preprocessed, tokenized and stemmed.
The preprocess Function uses SpaCy and Natural Language Toolkit (nltk) libraries to prepare the text for the pipeline. It removes punctuations, numbers and symbols from the clinical text. Then tokenization is applied and stemming is used to extract the root of each word.

2. Feature Engineering:
Similarity measures (features) in this pipeline include: 

a) Token based similarity measures 
Cao, MinHash, Johnson, Blue similarity measures are calculated using abydos.distance package libraries and Word_overlap (jaccard) similarity measure is calculated using the Word_overlap Function in the model.
Then, token based similarity measures are concatenated using token_based_sim Function.

b) Sequence based similarity measures
Bag , Levenshtein and Smith-Waterman measures are calculated using abydos.distance package libraries.
Then, sequnce based measures are concatenated using seq_based_sim Function.

c) Semantic based similarity measures
BioSentVec is calculated using get_biosentvec_similarity Function.
Word2Vec model is trained on BioWordVec data and is implemented in the pipeline by get_wmd_distance Function but it is not included in the final model. 
The Average vector similarity is calculated using get_mean_vector_similarity but it the measure is not used in the final model.
PubMedBERT, mediCal knOwledge embeDded tErm Representation (CODER) and BlueBERT models use pretrained Bert based models and similarity measures are calculated using  semantic_similarity_bert_based Function.
Then, semantic based measures are concatenated using semantic_based_sim Function.

Feature Concatenation:
All token based, sequnce based and semantic based features are concatnated in a dataframe using features_concat Function. The input is the text file path and output is a dataframe containing all the features (similarity measures) for the final model.

Feature Correlation:
The single_feature_correlation Function calculates features correlation with the gold standard score for the both validation and test data. The results are saved in .csv files.

3. Training, predicting and evaluating:

Training the model:
feature_selection_rf Function is used to train a Random Forest Model to predict a similarity scores using selected features. 
Linear Regression and Support Vector Machine models are also implemented but are not used in this model.
5-fold cross validation is used to generate training and validation data. 
GridSearchCV is used to tune hyperparameters for the Random Forest model.

Evaluating:
Mean Squared Error (MSE) is used to evaluate the model performance on test data. MSE is calculated in 5 different regions of similarity score (based on the gold standard score) to evaluate and assess the model performance for the lowest and the highest similar sentences and results are saved in mse.csv file and visulized on a bar chart.

The main Function inputs are the training, validation and test data file pathes.
Inside the main Function, feature_concat Function is called for each of the training, validation and test datasets to prepare fetures for the final model. These results are fed into the feature_selection_ef Function to train the model and then predict similarity measures using Random Forest model to generate results.





