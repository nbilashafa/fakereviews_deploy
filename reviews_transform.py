"""Transform module
"""
 
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "label"
FEATURE_KEY = "text_"

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
"yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
"them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
"was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
"but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
"through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
"again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
"most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
"just", "don", "should", "now"]
 
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    A function that preprocesses inputs by transforming feature and label keys.

    Parameters:
    inputs (dict): A dictionary containing feature and label keys.

    Returns:
    dict: A dictionary with transformed feature and label keys.
    """
    outputs = {}
    feature_key = tf.strings.lower(inputs[FEATURE_KEY])
    feature_key = tf.strings.regex_replace(feature_key, r"(?:<br />)", "")
    feature_key = tf.strings.regex_replace(feature_key, "n\'t", " not ")
    feature_key = tf.strings.regex_replace(feature_key, r"(?:\'ll |\'re |\'d |\'ve)", " ")
    feature_key = tf.strings.regex_replace(feature_key, r"\W+", " ")
    feature_key = tf.strings.regex_replace(feature_key, r"\d+", " ")
    feature_key = tf.strings.regex_replace(feature_key, r"\b[a-zA-Z]\b", " ")
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(feature_key, r'\b(' + r'|'.join(stopwords) + r')\b\s*', "")
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs