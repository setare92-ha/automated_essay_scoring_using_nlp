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
from spellchecker import SpellChecker

# Downloading resources for nltk
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger")

# Load and clean the concreteness dataset
# source: https://github.com/bodowinter/good_metaphors
concreteness_df = pd.read_csv("https://raw.githubusercontent.com/setare92-ha/automated_essay_scoring_using_nlp/refs/heads/main/clean_data/brysbaert_2014_concreteness.csv").dropna(subset=["Word"])

# Build the dictionary safely, ignoring missing or non-string values
concreteness_dict = {
    str(key).lower(): value
    for key, value in zip(concreteness_df["Word"], concreteness_df["Conc.M"])
    if pd.notnull(key)
}


# Preload stopwords and spellchecker
stopwords = set(nltk.corpus.stopwords.words("english"))
spell = SpellChecker()

# Function to get word concreteness
get_concreteness = np.vectorize(lambda word: concreteness_dict.get(word.lower(), 0))

# Function to compute LSA overlap
vectorizer = TfidfVectorizer()
svd = TruncatedSVD()

def compute_lsa_overlap(corpus):
    if len(corpus) < 2:
        return 0
    tfid_mat = vectorizer.fit_transform(corpus)
    lsa_vecs = svd.fit_transform(tfid_mat)
    similarity_vec = [
        cosine_similarity(lsa_vecs[i].reshape(1, -1), lsa_vecs[i + 1].reshape(1, -1))[0, 0]
        for i in range(lsa_vecs.shape[0] - 1)
    ]
    return np.mean(similarity_vec)

def extract_features_bulk(essays):
    docs = list(nlp.pipe(essays, batch_size=100, n_process=1))

    features = []

    for doc in docs:
        words = [token.text for token in doc if token.is_alpha]
        sentences = list(doc.sents)
        num_words = len(words)
        num_sentences = len(sentences)

        # Sentence lengths
        num_words_per_sentence = [len(list(sent)) for sent in sentences]

        # Length-based features
        avg_sentence_length = np.mean(num_words_per_sentence) if num_sentences else 0
        std_sentence_length = np.std(num_words_per_sentence, ddof=1) if num_sentences > 1 else 0

        # Word-level features
        num_syllables = [token._.syllables_count for token in doc if token.is_alpha]
        num_letters = [len(token.text) for token in doc if token.is_alpha]

        avg_syllable_count = np.mean(num_syllables) if num_syllables else 0
        std_syllable_count = np.std(num_syllables, ddof=1) if len(num_syllables) > 1 else 0

        avg_letter_count = np.mean(num_letters) if num_letters else 0
        std_letter_count = np.std(num_letters, ddof=1) if len(num_letters) > 1 else 0

        # Readability
        flesch_reading_ease = textstat.flesch_reading_ease(doc.text)
        flesh_kincaid_grade = textstat.flesch_kincaid_grade(doc.text)
        smog_index = textstat.smog_index(doc.text)

        # Lexical diversity
        lexical_diversity = len(set(words)) / num_words if num_words else 0

        # Stopwords, pronouns, verbs
        num_stopwords = sum(1 for word in words if word.lower() in stopwords) / num_words
        num_pronouns = sum(1 for token in doc if token.pos_ == "PRON") / num_words
        num_verbs = sum(1 for token in doc if token.pos_ == "VERB") / num_words

        # Syntactic simplicity
        avg_tree_depth = sum(len(list(token.ancestors)) for token in doc) / num_words

        # Word concreteness
        avg_concreteness = np.mean(get_concreteness(words)) if num_words else 0

        # Misspellings
        misspelled = spell.unknown(words)
        num_misspelled = len(misspelled) / num_words

        # LSA Overlap
        sents_text = [sent.text.strip() for sent in sentences if sent.text.strip()]
        lsa_overlap_avg = compute_lsa_overlap(sents_text)

        # Store features
        features.append([
            num_words, num_sentences, avg_sentence_length, std_sentence_length,
            avg_syllable_count, std_syllable_count, avg_letter_count, std_letter_count,
            flesch_reading_ease, flesh_kincaid_grade, smog_index,
            lexical_diversity, num_stopwords, num_pronouns, num_verbs,
            avg_tree_depth, avg_concreteness, num_misspelled, lsa_overlap_avg
        ])

    # Convert to DataFrame
    feature_names = [
        "num_words", "num_sentences", "avg_sentence_length", "std_sentence_length",
        "avg_syllable_count", "std_syllable_count", "avg_letter_count", "std_letter_count",
        "flesch_reading_ease", "flesh_kincaid_grade", "smog_index",
        "lexical_diversity", "num_stopwords", "num_pronouns", "num_verbs",
        "avg_tree_depth", "avg_concreteness", "num_misspelled", "lsa_overlap_avg"
    ]

    return pd.DataFrame(features, columns=feature_names)
