import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re  
from abydos.distance import PositionalQGramDice as pqgd , PositionalQGramJaccard as pqgj, PositionalQGramOverlap as pqgo, Cao , MinHash, Johnson, BLEU, Cosine, Dice
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import swalign
from abydos.distance import Bag
from abydos.distance import sim_bag , NeedlemanWunsch as nw , SmithWaterman as SW, Gotoh , sim_levenshtein
import enchant
from numpy import dot
from numpy.linalg import norm
import sent2vec 
from scipy.spatial import distance
import gensim
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from gensim import models
import fasttext
from sklearn.metrics import mean_squared_error
import transformers
from transformers import AutoModel , AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from scipy.stats import pearsonr
from sklearn import tree
from datetime import datetime



#import data
#train= pd.read_csv('/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/train.tsv', sep='\t')
#valid= pd.read_csv('/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/valid.tsv', sep='\t')
#test= pd.read_csv('/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/test.tsv', sep='\t')
#trainValid = train.append(valid, ignore_index=True)
#trainValid.to_csv("trainValid.tsv", sep="\t")
#print(trainValid)

def preprocess(text):
      text = text.replace('/', ' / ') #separate words with / in between
      text = text.replace('.-', ' .- ')
      text = text.replace('.', ' . ')
      text = text.replace('\'', ' .\' ')
      text = re.sub(r'[^\w\s]', '', text) # remove punctuations
      text = re.sub(r'\d+','',text)# remove numbers
      text = text.lower() # lower case 
      nlp = spacy.load('en_core_web_sm')

      tokens = nlp(text) 
      token_list =[]
      for token in tokens:
          token_list.append(token.text)
      cleaned_text=[]
      for word in token_list:
          lexeme=nlp.vocab[word]
          if lexeme.is_stop == False:
            word = PorterStemmer().stem(word)
            cleaned_text.append(word)

      return  (" ").join(cleaned_text)
   
#simple_word_overlap
def word_overlap(text1,text2):
  words_lists = [text.split() for text in [text1, text2]]
  words1, words2 = words_lists
  #word lists to word sets
  words_sets = [set(words) for words in words_lists]
  # simple word overlap
  words_set1 = words_sets[0]
  for i, words_set in enumerate(words_sets[1:], 2):
      shared_words = words_set1 & words_set
      total_words = words_set1 | words_set
      overlap = len(shared_words)/len(total_words)
  return overlap


def jaccard_sim(str1, str2):
  a=set(str1.split())
  b=set(str2.split())
  c=a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))

#cosine similarity using term frequency
def cosine_sim(text1, text2):
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.metrics.pairwise import cosine_similarity
  # Create the Document Term Matrix
  count_vectorizer = CountVectorizer(stop_words='english')
  count_vectorizer = CountVectorizer()
  text=[text1, text2]
  sparse_matrix = count_vectorizer.fit_transform(text)
  doc_term_matrix = sparse_matrix.todense()
  return cosine_similarity(doc_term_matrix, doc_term_matrix)

#tanimoto similarity (cosine using binary values) token_based
def tanimoto_sim(text1, text2):
  from sklearn.preprocessing import Binarizer
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.metrics.pairwise import cosine_similarity
  count_vectorizer = CountVectorizer(stop_words='english')
  count_vectorizer = CountVectorizer(binary=True)
  text=[text1, text2]
  sparse_matrix = count_vectorizer.fit_transform(text)
  doc_term_matrix = sparse_matrix.todense()
  return cosine_similarity(doc_term_matrix, doc_term_matrix)

#tfidf token_based
def tfidf_sim(text1, text2):
    import nltk, string
    from sklearn.feature_extraction.text import TfidfVectorizer
    text=[text1, text2]
    vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
    tfidf = vect.fit_transform(text)                                                                                                                                                                                                                       
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity

# token-based similarity
#Qgram dice
#pqgd().sim()

#Qgram jaccard
#pqgj().sim()

#QgramOverlap
#pqgo().sim()

#Cao distance similarity
#Cao().sim()

#MinHash similarity
#MinHash().sim()

#Johnson similarity
#Johnson().sim()

#BLEU similarity
#BLEU().sim()

#Cosine similarity
#Cosine().sim()

#Dice similarity
#Dice().sim()



# Wunsch similarity , paiwise alignment (seq alignment), seq-based
#alignments = pairwise2.align.globalxx(text1, text2)
#print(format_alignment(*alignments[0]))


#Waterman Similarity, seq-based
#match = 2
#mismatch = -1
#scoring = swalign.NucleotideScoringMatrix(match, mismatch)

#sw = swalign.LocalAlignment(scoring)  
#alignment = sw.align(text1, text2)
#alignment.dump()
#print(alignment)


#seq_based similarity
#Bag of words
#Bag().sim(text1, text2)

#Needleman-wunsch
#nw().sim(text1, text2)

#Smith Waterman
#SW().sim(text1, text2)

#Gotoh
#Gotoh().sim(text1, text2)

#levenshtein 
#enchant.utils.levenshtein(text1, text2)
#sim_levenshtein(text1, text2)



#semantic_based similarity BlueBERT

def semantic_similarity_bert_based(text1, text2, model, tokenizer):
  inputs_1 = tokenizer(text1, return_tensors='pt')
  inputs_2 = tokenizer(text2, return_tensors='pt')
  sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
  sent_2_embed = np.mean(model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)

  similarities = dot(sent_1_embed, sent_2_embed)/(norm(sent_1_embed)* norm(sent_2_embed))
  return similarities

                    
    
blue_bert_model = AutoModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
blue_bert_tokenizer = AutoTokenizer.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
  
#semantic_based sim BioSentVec 
model_bsv = sent2vec.Sent2vecModel()
try:
  model_bsv.load_model('/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/models/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')  # load from the path model is saved at
except Exception as e:
  print(e)
print('BioSentVec model successfully loaded')
def get_biosentvec_similarity(text1, text2):
    sentence_v1 = model_bsv.embed_sentence(text1)
    sentence_v2 = model_bsv.embed_sentence(text2)
    cosine_sim = 1-distance.cosine(sentence_v1, sentence_v2)
    return cosine_sim
 

# semantic based similarity wmd, and the performance not good, cannot detec dissimilar distances

#Train Word2Vec on BioWordVec
#model_wordvec = Word2Vec('/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/models/BioWordVec_PubMed_MIMICIII_d200.bin')
#vocab = model_wordvec.wv.key_to_index
def get_wmd_distance(text1, text2):
  distance = model_wordvec.wv.wmdistance(text1, text2)
  return distance
#semantic similarity through avg vector & cosine similarity
def get_mean_vector_similarity(model_wordvec, text1, text2):
    # remove out-of-vocabulary words
    text1 = [word for word in text1 if word in vocab]
    ans1 = np.mean(model_wordvec.wv[text1], axis=0)
    text2 = [word for word in text2 if word in vocab]
    ans2 = np.mean(model_wordvec.wv[text2], axis=0)
    ans  = dot(ans1, ans2)/(norm(ans1)*norm(ans2))
    return ans

# semantic through PubMedBERT
pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

#semantic similarity mediCal knOwledge embeDded tErm Representation (CODER)
coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

#token based features
def token_based_sim (text1, text2, remove_prefix = False):
  similarity_measures = [word_overlap(text1,text2), Cao().sim(text1, text2), MinHash().sim(text1, text2), Johnson().sim(text1, text2), BLEU().sim(text1, text2), Cosine().sim(text1, text2)]
              
  token_based_features = []
  for i in range(len(similarity_measures)):
    feature = similarity_measures[i]
    token_based_features.append(feature*5)
  return token_based_features


#seq_based features
def seq_based_sim (text1, text2):
  similarity_measures = [Bag().sim(text1, text2) , sim_levenshtein(text1, text2), SW().sim(text1, text2)]                        
  seq_based_features = []
  for i in range(len(similarity_measures)):
    feature = similarity_measures[i]
    seq_based_features.append(feature*5)
  return seq_based_features

#semantic based features
def semantic_based_sim(text1, text2):
  similarity_measures = [get_biosentvec_similarity(text1, text2), 
                         semantic_similarity_bert_based(text1, text2, blue_bert_model, blue_bert_tokenizer), 
                         semantic_similarity_bert_based(text1, text2, pubmed_bert_model, pubmed_bert_tokenizer), 
                         semantic_similarity_bert_based(text1, text2, coder_model, coder_tokenizer)]
  semantic_based_features = []
  for i in range(len(similarity_measures)):
    feature = similarity_measures[i]
    semantic_based_features.append(feature*5)
  return semantic_based_features

#GridSearchCV
def rf_best_params(training_data):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    from sklearn.model_selection import GridSearchCV
    rf = RandomForestRegressor()
    param_grid = {
      'bootstrap' : [True],
      'max_depth' : [8, 10, 12, 14],
      'n_estimators' : [100, 200, 300],
    }
    rf_Grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv=5, verbose=2, n_jobs=-1)
    X = training_data.iloc[:,:-1]
    y = training_data.iloc[:,-1]
    rf_Grid.fit(X, y) # Train the model on training data with all features
    return rf_Grid.best_params_


#feature selection & prediction using Random forest
def feature_selection_rf(train_data, valid_data, test_data):
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    from sklearn.model_selection import GridSearchCV
 
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, bootstrap=True, random_state=90)
    lr = LinearRegression()
    svr = SVR(kernel='linear') 
  
    #data for k fold
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    test_x = test_data.iloc[:,:-1]
    test_y = test_data.iloc[:,-1]
    valid_x = valid_data.iloc[:,:-1]
    valid_y = valid_data.iloc[:,-1]
    test_df= pd.read_csv('/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/test.tsv', sep='\t', header=0, names=['query', 'subject', 'score', 'id'])
    test_id = test_df.iloc[:,-1]


  
    #5 fold cross validation on training and velidation set
    kf = KFold(n_splits=5)
    corr=[]
    for train_index , valid_index in kf.split(X_train):
        X_train , X_valid = X_train.iloc[train_index,:], X_train.iloc[valid_index,:]
        y_train , y_valid = y_train[train_index] , y_train[valid_index]
    
        #fit the model in loop for kfold, indent with k fold
        rf.fit(X_train, y_train) # Train the model on training data with all features
        #lr.fit(X_train, y_train)
        #svr.fit(X_train, y_train)
        #test on validation data, indent for k fold
        val_pred = rf.predict(X_valid)
        #val_pred = lr.predict(valid_x)
        #val_pred = svr.predict(X_valid)
        #evaluate pearson correlation on validation data prediction by adding features (to find the most important ones)
        eval_metric_valid = pearsonr(y_valid,val_pred)[0]
        corr.append(eval_metric_valid)
      #k fold avg and std, out of loop
    avg = np.mean(corr)
    standard_dev = np.std(corr)
        # Print out the pearson r
      #print('Average and std of Pearson Correlation of features on 5-fold validation data:', avg, standard_dev)
        #print("prediction and label" , val_pred, val_labels)
    # mse and pearson for test set
    start_time = datetime.now()
    test_pred = rf.predict(test_x)
    mse = mean_squared_error(test_y, test_pred)
    end_time = datetime.now()
    #print('Duration: {}'.format(end_time - start_time))

    #test_pred = lr.predict(test_x)
    #test_pred = svr.predict(test_x)

    data = {'sentence_pair_id': test_id,
            'gold_standard_score':test_y,
            'model_prediction': test_pred,
            'mse' : mse
            }
    model_result_df = pd.DataFrame(data)
    model_result_df.to_csv(r'RFmodel_result.csv')


    
    test_pear = pearsonr(test_y,test_pred)[0]
  #print ('Pearson correlation on test data:', test_pear)
  # mse and pearson for validation data
    valid_pred = rf.predict(valid_x)
    mse_valid = mean_squared_error(valid_y , valid_pred)
    pearsonr_valid = pearsonr(valid_y , valid_pred)[0]
  #print ('Pearson correlation and mse on original validation data:', mse_valid , pearsonr_valid)
    import csv
    model_correlation=[pearsonr_valid, test_pear]
    file = open('model_correlation_token_basedPlusSeq_based.csv', 'w', newline ='')
  
    with file:
    # identifying header  
        header = ['validation', 'test']
        writer = csv.DictWriter(file, fieldnames = header)
      
    # writing data row-wise into the csv file
        writer.writeheader()
        writer.writerow({'validation' : pearsonr_valid , 'test': test_pear})

  #categorizing labels to 5 regions
    test1 , test2, test3, test4, test5 = test_data.query('label<1').copy(), test_data.query('1<label<=2').copy(), test_data.query('2<label<=3').copy(),test_data.query('3<label<=4').copy(), test_data.query('4<label<=5').copy()
    test1_x = test1.iloc[:,:-1]
    test1_y = test1.iloc[:,-1]

    test1_pred = rf.predict(test1_x)
    mse1 = mean_squared_error(test1_y, test1_pred)

    test2_x = test2.iloc[:,:-1]
    test2_y = test2.iloc[:,-1]
    test2_pred = rf.predict(test2_x)
    mse2 = mean_squared_error(test2_y, test2_pred)

    test3_x = test3.iloc[:,:-1]
    test3_y = test3.iloc[:,-1]
    test3_pred = rf.predict(test3_x)
    mse3 = mean_squared_error(test3_y, test3_pred)

    test4_x = test4.iloc[:,:-1]
    test4_y = test4.iloc[:,-1]
    test4_pred = rf.predict(test4_x)
    mse4 = mean_squared_error(test4_y, test4_pred)

    test5_x = test5.iloc[:,:-1]
    test5_y = test5.iloc[:,-1]
    test5_pred = rf.predict(test5_x)
    mse5 = mean_squared_error(test5_y, test5_pred)
  
    mse = [mse1, mse2, mse3, mse4, mse5]
    print(mse)
    file = open('mse0.csv', 'w', newline ='')
  
    with file:
    # identifying header  
        header = ["(0,1]", "(1,2]", "(2,3]", "(3,4]", "(4,5]"]
        writer = csv.DictWriter(file, fieldnames = header)
      
    # writing data row-wise into the csv file
    writer.writeheader()
    writer.writerow({'(0,1]' : mse1 , '(1,2]' : mse2, '(2,3]' : mse3, '(3,4]' : mse4, '(4,5]' : mse5})



    ##bar chart for mse
    bar_data = {'(0,1]':mse1, '(1,2]':mse2, '(2,3]':mse3, '(3,4]':mse4, '(4,5]':mse5}
    mse = list(bar_data.keys())
    values = list(bar_data.values())
    fig = plt.figure(figsize=(10,5))
    ## creating the bar plot
    plt.bar(mse, values, color ='maroon', width = 0.4)
    fig.savefig('mse_bar.png') 
    plt.xlabel("Mean Squared Error")
    plt.show()

  # visualizing a decision tree
    fig = plt.figure()
    fn=["word_overlap/jaccard", "Cao", "MinHash", "Johnson","Bleu", "Cosine", "Bag", "lev", "SW", "biosentvec","bluebert", "pubmed bert", "CODER"]

    _  =tree.plot_tree(rf.estimators_[0], feature_names = fn, filled = True, impurity = True, rounded = True)
    fig.savefig('tree.png') 



#https://medium.com/analytics-vidhya/feature-selection-techniques-2614b3b7efcd




#feature concatation

def features_concat(file_path):
  df=pd.read_csv(file_path, sep='\t', header=0, names=['query', 'subject', 'score', 'id'])
  df_features = []
  

  for _ , row in df.iterrows():
        features = []
        query = preprocess(row.query) 
        subject = preprocess(row.subject)
        label = row.score
       
        token_based_features = token_based_sim(query, subject)
        seq_based_features = seq_based_sim(query, subject)
        semantic_based_features = semantic_based_sim(query, subject)

        features = np.concatenate((token_based_features, seq_based_features, semantic_based_features, label), axis=None)
        
        df_features.append(features)
        
  df_features = pd.DataFrame(df_features, columns=["word_overlap/jaccard", "Cao", "MinHash", "Johnson","Bleu", "Cosine", "Bag", "lev", "SW", "biosentvec","bluebert", "pubmed bert", "CODER", "label"])
    
  return df_features

def single_feature_correlation(valid_data, test_data):
    test_label = test_data.iloc[:,-1]
    valid_label = valid_data.iloc[:,-1]
    test_data = test_data.iloc[:,:-1]
    valid_data = valid_data.iloc[:,:-1]

    valid_correlation =[]
    test_correlation =[]
    correlation =[]
    for column in valid_data:
        valid_corr = pearsonr(valid_data[column].values, valid_label)[0]
        valid_correlation.append(valid_corr)
    for column in test_data:
        test_corr = pearsonr (test_data[column].values, test_label)[0]
        test_correlation.append(test_corr)
    df_valid_correlation =pd.DataFrame(valid_correlation, index=["word_overlap/jaccard_valid_correlation", "Cao_valid_correlation", "MinHash_valid_correlation", "Johnson_valid_correlation", "Bleu_valid_correlation", "Cosine_valid_correlation", 
                                                     "Bag_valid_correlation", "lev_valid_correlation", "SW_valid_correlation" ,
                                                      "biosentvec_valid_correlation","bluebert_valid_correlation", "pubmed bert_valid_correlation", "CODER_valid_correlation"])
                                                     
    df_valid_correlation = df_valid_correlation.T
    df_valid_correlation.to_csv(r'valid_feature_correlation.csv', index=False)

    df_test_correlation =pd.DataFrame(test_correlation, index=["word_overlap/jaccard_test_correlation", "Cao_test_correlation", "MinHash_test_correlation", "Johnson_test_correlation",  "Bleu_test_correlation",  "Cosine_test_correlation", 
                                                     "Bag_test_correlation","lev_test_correlation", "SW_test_correlation", 
                                                     "biosentvec_test_correlation", "bluebert_test_correlation", "pubmed bert_test_correlation", "CODER_test_correlation"])
    df_test_correlation = df_test_correlation.T
    df_test_correlation.to_csv(r'test_feature_correlation.csv', index=False)
        
        



def main(train_file_path, valid_file_path, test_file_path):
  #trainValid_data = features_concat(trainValid_file_path)
  #print(trainValid_data)
  train_data = features_concat(train_file_path)
  valid_data = features_concat(valid_file_path)
  test_data = features_concat(test_file_path)
  #single_feature_correlation(valid_data, test_data)
  feature_selection_rf(train_data, valid_data, test_data)

#train_file_path = '/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/train.tsv'
valid_file_path = '/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/valid.tsv'
trainValid_file_path = '/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/trainValid.tsv'
test_file_path = '/panfs/pan1/bionlp/lulab/qingyu/elaheh/SentenceSimilarity/test.tsv'
main(trainValid_file_path, valid_file_path, test_file_path)
#trainValid_data = features_concat(trainValid_file_path)
#print(rf_best_params(trainValid_data))
#train_data = features_concat(train_file_path)
#valid_data = features_concat(valid_file_path)
#test_data = features_concat(test_file_path)
#single_feature_correlation(valid_data, test_data)
#feature_selection_rf(trainValid_data, test_data, valid_data)











    

