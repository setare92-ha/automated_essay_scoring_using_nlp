import pandas as pd
import textstat
import spacy
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from spacy_syllables import SpacySyllables

# Downloading resources for nltk
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger")


def get_concreteness(word):
    df_concreteness = pd.read_csv("./data/brysbaert_2014_concreteness.csv")
    dict_concreteness = {
        key: value
        for key, value in zip(df_concreteness["Word"], df_concreteness["Conc.M"])
    }
    return dict_concreteness.get(word.lower(), 0)


def extract_nouns(sentence):
    sent_nouns = {token.lemma_.lower() for token in sentence if token.pos_ == "NOUN"}
    return sent_nouns


def extract_arguments(sentence):
    sent_args = {
        token.lemma_.lower()
        for token in sentence
        if token.dep_ in ["nsubj", "nsubjpass", "dobj", "pobj", "iobj"]
        or token.pos_ == "PRON"
    }
    return sent_args


def extract_lemmas(sentence):
    sent_lemmas = {token.lemma_ for token in sentence}
    return sent_lemmas


def compute_lsa_overlap(corpus):
    vectorizer = TfidfVectorizer()
    tfid_mat = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD()
    lsa_vecs = svd.fit_transform(tfid_mat)
    similarity_vec = [
        cosine_similarity(lsa_vecs[i].reshape(1, -1), lsa_vecs[i + 1].reshape(1, -1))[
            0, 0
        ]
        for i in range(lsa_vecs.shape[1])
    ]
    lsa_overlap_avg = np.mean(similarity_vec)
    lsa_overlap_std = np.std(similarity_vec, ddof=1)

    return lsa_overlap_avg, lsa_overlap_std


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
        num_words_per_sentence.append(
            len([token.text for token in sentence if token.is_alpha])
        )

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
    flesh_kincaid_grade = textstat.flesch_kincaid_grade(essay)
    smog_index = textstat.smog_index(essay)

    #####################################
    ######### Lexical diversity #########
    #####################################
    unique_words = len(set(words))
    lexical_diversity = unique_words / num_words if num_words else 0

    ##############################################################
    ######### Text easability principal component scores #########
    ##############################################################
    # cohesion and discourse markers
    stopwords = set(nltk.corpus.stopwords.words("english"))
    num_stopwords = sum(1 for token in words if token.lower() in stopwords) / num_words

    ## narrativity metrics
    num_pronouns = (
        sum(1 for token in doc if token.pos_ == "PRON") / num_words if num_words else 0
    )  # normalized ratio of pronouns
    num_verbs = (
        sum(1 for token in doc if token.pos_ == "VERB") / num_words if num_words else 0
    )  # normalized ratio of verbs
    # number of familiar words: difficult to get at the moment

    ###############################
    ######### Connectives #########
    ###############################
    connectives_english = set(
        [
            "and",
            "also",
            "moreover",
            "furthermore",
            "besides",
            "additionally",
            "but",
            "however",
            "on the other hand",
            "although",
            "whereas",
            "nevertheless",
            "nonetheless",
            "because",
            "therefore",
            "thus",
            "so",
            "before",
            "after",
            "meanwhile",
            "if",
            "unless",
            "for example",
            "for instance",
            "in conclusion",
            "overall",
            "likewise"
        ]
    )

    connectives = (
        sum(1 for token in doc if token.text.lower() in connectives_english) / num_words
    )

    ## syntactic simplicity
    # avg_sentence_length is one
    avg_tree_depth = (
        sum(len(list(token.ancestors)) for token in doc) / num_words if num_words else 0
    )

    ## word concreteness
    words_concreteness = [get_concreteness(word) for word in words]
    avg_concreteness = sum(words_concreteness) / len(words_concreteness)

    ########################################
    ######### Referential Cohesion #########
    ########################################
    ## noun overlap, adjacent sentences, binary, mean
    noun_overlaps = []
    for i in range(num_sentences - 1):
        curr_set = extract_nouns(sentences[i])
        next_set = extract_nouns(sentences[i + 1])
        noun_overlaps.append(1 if curr_set.intersection(next_set) else 0)

    avg_adj_noun_overlaps = sum(noun_overlaps) / len(noun_overlaps)

    ## argument overlap, adjacent sentences, binary, mean
    arg_overlaps = []
    for i in range(num_sentences - 1):
        curr_set = extract_arguments(sentences[i])
        next_set = extract_arguments(sentences[i + 1])
        arg_overlaps.append(1 if curr_set.intersection(next_set) else 0)

    avg_adj_arg_overlaps = sum(arg_overlaps) / len(arg_overlaps)

    ## stem/lemma overlap, adjacent sentences, binary, mean
    lemma_overlaps = []
    for i in range(num_sentences - 1):
        curr_set = extract_lemmas(sentences[i])
        next_set = extract_lemmas(sentences[i + 1])
        lemma_overlaps.append(1 if curr_set.intersection(next_set) else 0)
    avg_adj_lemma_overlaps = sum(lemma_overlaps) / len(lemma_overlaps)

    ##################################################
    ######### Latent Semantic Analysis (LSA) #########
    ##################################################
    # LSA overlap, adjacent sentences, mean
    sents_stripped = [sent.text.strip() for sent in sentences if sent.text.strip()]
    lsa_overlap_avg, lsa_overlap_std = compute_lsa_overlap(sents_stripped)

    ######### syntactic complexity #########
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

    avg_words_before_verb = (
        sum(num_words_before_verb) / len(num_words_before_verb)
        if len(num_words_before_verb)
        else 0
    )

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
        "flesh_kincaid_grade": flesh_kincaid_grade,
        "smog_index": smog_index,
        "lexical_diversity": lexical_diversity,
        "num_stopwords": num_stopwords,
        "num_pronouns": num_pronouns,
        "num_verbs": num_verbs,
        "avg_tree_depth": avg_tree_depth,
        "avg_concreteness": avg_concreteness,
        "avg_adj_noun_overlaps": avg_adj_noun_overlaps,
        "avg_adj_arg_overlaps": avg_adj_arg_overlaps,
        "avg_adj_lemma_overlaps": avg_adj_lemma_overlaps,
        "lsa_overlap_avg": lsa_overlap_avg,
        "lsa_overlap_std": lsa_overlap_std,
        "connectives": connectives,
        "avg_words_before_verb": avg_words_before_verb,
    }
