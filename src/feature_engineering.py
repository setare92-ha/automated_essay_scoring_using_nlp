import pandas as pd
import textstat
import spacy
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from spacy_syllables import SpacySyllables

# Downloading resources for nltk
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger")


def extract_coh_metrix_features(essay: str) -> dict:
    """
    The function extracts Coh-Metrix-like features.
    Arguments:
        essay: string containing the essay text
    Outouts:
        A dictionary containing Coh-Metrix-like metrics as follows:
    """
    doc = nlp(essay)
    words = [token.text for token in doc if token.is_alpha]
    sentences = list(doc.sents)
    
    ###############################
    ######### Descriptive #########
    ###############################
    
    ## number of words and sentences
    num_words = len(words)
    num_sentences = len(sentences)

    ## sentence length, avg & std
    num_words_per_sentence = []
    for sentence in sentences:
            num_words_per_sentence.append(len([token.text for token in sentence if token.is_alpha]))
    
    if num_sentences:
        avg_sentence_length = np.mean(num_words_per_sentence)
        std_sentence_length = np.std(num_words_per_sentence, ddof=1)
    else:
        avg_sentence_length = 0
        std_sentence_length = 0
    
    ## word length, number of syllables, avg & std
    ## word length, number of letters, avg & std
    num_syllables_per_word = []
    num_letters_per_word = []
    words_token = [token for token in doc if token.is_alpha]

    for token in words_token:
         num_syllables_per_word.append(token._.syllables_count)
         num_letters_per_word.append(len(token.text))

    
    if num_words:
        avg_syllable_count = np.mean(num_syllables_per_word)
        std_syllable_count = np.std(num_syllables_per_word, ddof=1)

        avg_letter_count = np.mean(num_letters_per_word)
        std_letter_count = np.std(num_letters_per_word, ddof=1)
    else:
        avg_syllable_count = 0
        std_syllable_count = 0

        avg_letter_count = 0
        std_letter_count = 0
    
    
    # Readability metrics
    flesch_reading_ease = textstat.flesch_reading_ease(essay)
    smog_index = textstat.smog_index(essay)

    # Lexical diversity
    unique_words = len(set(words))
    lexical_diversity = unique_words/num_words if num_words else 0

    ##############################################################
    ######### Text easability principal component scores #########
    ##############################################################
    # cohesion and discourse markers
    stopwords = set(nltk.corpus.stopwords.words("english"))
    num_stopwords = sum(1 for token in words if token.lower() in stopwords)/num_words
    
    ## narrativity metrics
    num_pronouns = sum(1 for token in doc if token.pos_=="PRON")/num_words if num_words else 0 # normalized ratio of pronouns
    num_verbs = sum(1 for token in doc if token.pos_=="VERB")/num_words if num_words else 0 # normalized ratio of verbs
    # number of familiar words: difficult to get at the moment
    connectives = sum(1 for token in doc if token.text.lower() in ["however", "therefore", "moreover", "thus", "furthermore", "and", "but", "although"])
    
    ## syntactic simplicity
    # avg_sentence_length is one
    avg_tree_depth = sum(len(list(token.ancestors)) for token in doc) / num_words if num_words else 0

    ## word concreteness
    df_concreteness = pd.read_csv('./data/brysbaert_2014_concreteness.csv')
    dict_concreteness = {key:value for key,value in zip(df_concreteness['Word'], df_concreteness['Conc.M'])}

    def get_concreteness(word):
        return dict_concreteness.get(word.lower(), 0)
    
    words_concreteness = [get_concreteness(word) for word in words]
    avg_concreteness = sum(words_concreteness)/len(words_concreteness)

    ## number of overlapping words in consecutive sentences
    num_overlaps = []
    for i in range(num_sentences-1):
        curr_set = set([token.lemma_.lower() for token in sentences[i]])
        next_set = set([token.lemma_.lower() for token in sentences[i+1]])
        num_overlaps.append(len(curr_set.intersection(next_set)))
    avg_overlaps = sum(num_overlaps)/len(num_overlaps)
    
    # syntactic complexity
    num_words_before_verb = []
    for sentence in sentences:
        count = 0
        verb_found = False

        for token in sentence:
            if token.pos_ == "VERB":
                verb_found = True
                break
            count += 1
        
        if verb_found:
            num_words_before_verb.append(count)
    
    avg_words_before_verb = sum(num_words_before_verb)/len(num_words_before_verb) if len(num_words_before_verb) else 0

    return {
        # descriptive
        "num_words": num_words,
        "num_sentences": num_sentences,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length,
        "avg_syllable_count": avg_syllable_count,
        "std_syllable_count": std_syllable_count,
        "avg_letter_count": avg_letter_count,
        "std_letter_count": std_letter_count,
        "flesch_reading_ease": flesch_reading_ease,
        "smog_index": smog_index,
        "lexical_diversity": lexical_diversity,
        "num_stopwords": num_stopwords,
        "num_pronouns": num_pronouns,
        "num_verbs": num_verbs,
        "avg_tree_depth": avg_tree_depth,
        "avg_concreteness": avg_concreteness,
        "avg_overlaps": avg_overlaps,
        "connectives": connectives,
        "avg_words_before_verb": avg_words_before_verb
    }