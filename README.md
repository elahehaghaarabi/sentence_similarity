# Sentence Similarity
The sentence_similarity.py contsains the code to capture sentence similarity of sentence pairs. 
Preprocess Function:
This function uses SpaCy and nltk libraries to prepare the text for the pipeline. It removes punctuations, numbers and symbols from the clinical text. Then tokenization is applied and stemming is used to extract the root of each word.
Similarity measures in this pipeline include: 
1) Token based similarity measures 
Cao, MinHash, Johnson, Blue similarity measures are calculated using abydos.distance package and Word_overlap measure is calculated using the same Word_overlap function.
Then token based similarity measures are concatenated using token_based_sim Function.
2) Sequence based similarity measures
Bag , Levenshtein and Smith-Waterman measures are calculated using abydos.distance module.
Then sequnce based measures are concatenated using seq_based_sim Function.
3) Semantic based similarity measures
BioSentVec is calculated using get_biosentvec_similarity Function.
Word2Vec model is trained on BioWordVec data and is implemented in the pipeline by get_wmd_distance Function but is not used in the final model. 
The Average vector similarity is calculated using get_mean_vector_similarity but the measure is not used in the final model.
PubMedBERT, mediCal knOwledge embeDded tErm Representation (CODER) and BlueBERT models use pretrained Bert based models and similarity measure is calculated using  semantic_similarity_bert_based Function.
Semantic based measures are concatenated using semantic_based_sim Function.

Feature Concatenation:
All token based, sequnce based and semantic based features are concatnated in a dataframe using features_concat Function. The input is the text file path and output is a dataframe containing all the features (similarity measures) for the final model.

Feature Correlation:
The single_feature_correlation Function calculates the model predicted output correlation with the gold standard score for the both validation and test data. The results are saved in .csv files.

Training the model:
feature_selection_rf Function is used to train a Random Forest Model to predict a similarity scores using selected features. 
Linear Regression and Support Vector Machine models are also implemented but are not used in this model.
5-fold cross validation is used to generate training and validation data. 
GridSearchCV is used for hyperparameter tunning.
Mean Squared Error is used to evaluate the model performance on test data. MSE is calculated in 5 different regions of similarity to check the model performance for the lowest and highest similar sentences and the results are both saved in mse.csv file and visulized in a bar chart.

The main Function inputs are the training, validation and test data file pathes.
Insode the main Function feature_concat Function is called for each of the datasets to prepare fetures for the model. Thes results are fed into the feature_selection_ef model to train and then predict similarity measures using Random Forest model and generate outputs.





