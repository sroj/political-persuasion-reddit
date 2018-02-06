import argparse
import csv
import functools
import json
import re
import logging

import numpy as np

prefix_wordlist = ''
wordlists_dir = prefix_wordlist + 'Wordlists/'

filename_norm_bristol_gilhooly_logie = wordlists_dir + "BristolNorms+GilhoolyLogie.csv"
filename_norm_warringer = wordlists_dir + "Ratings_Warriner_et_al.csv"

regex_tokenizer = re.compile(r"[^\S\n]*(\n)+[^\S\n]*|[^\S\n]+")


def read_bgl_norms():
    with open(filename_norm_bristol_gilhooly_logie, newline='') as csv_file:
        csv_dict = csv.DictReader(csv_file, restkey="rest")

        bgl_norm_dict = dict()

        for row in csv_dict:
            word = row['WORD']

            if not word:
                logging.warning("the word field was missing, skipping...")
                continue
                
            aoa = int(row['AoA (100-700)'])
            img = int(row['IMG'])
            fam = int(row['FAM'])
            bgl_norm_dict[word] = [aoa, img, fam]

    return bgl_norm_dict


bgl_norms = read_bgl_norms()


def read_files_by_line(directory, files):
    lines = list()

    for file in files:
        with open(directory + file) as f:
            for line in f:
                lines.append(line.strip())

    return lines


def read_first_person_pronouns():
    files = ["First-person"]
    return set(read_files_by_line(wordlists_dir, files))


def read_second_person_pronouns():
    files = ["Second-person"]
    return set(read_files_by_line(wordlists_dir, files))


def read_third_person_pronouns():
    files = ["Third-person"]
    return set(read_files_by_line(wordlists_dir, files))


def read_slang_acronyms():
    files = ["Slang"]
    return set(read_files_by_line(wordlists_dir, files))


first_person_pronouns = read_first_person_pronouns()
second_person_pronouns = read_second_person_pronouns()
third_person_pronouns = read_third_person_pronouns()
slang_acronyms = set(filter(None, read_slang_acronyms()))

# First person pronouns
fpp_alternation = "|".join(first_person_pronouns)
fpp_group = r"(?:{})".format(fpp_alternation)
regex_fpp = re.compile(r"^{0}/[^\s/]+$".format(fpp_group), re.IGNORECASE)

# Second person pronouns
spp_alternation = "|".join(second_person_pronouns)
spp_group = r"(?:{})".format(spp_alternation)
regex_spp = re.compile(r"^{0}/[^\s/]+$".format(spp_group), re.IGNORECASE)

# Third person pronouns
tpp_alternation = "|".join(third_person_pronouns)
tpp_group = r"(?:{})".format(tpp_alternation)
regex_tpp = re.compile(r"^{0}/[^\s/]+$".format(tpp_group), re.IGNORECASE)

# Coordinating conjunctions
regex_cc = re.compile(r"^[^\s/]+/CC$")

# Past tense verbs
regex_ptv = re.compile(r"^[^\s/]+/VBD$")

# Future tense verbs
regex_ftv = re.compile(r"(?:\s+|^)(?:will|'ll)/[^\s/]+(?:\s+|$)", re.IGNORECASE)
regex_gonna_vb = re.compile(r"(?:\s+|^)gonna/[^\s/]+\s+\S+/VB[DGNPZ]?(?:\s+|$)", re.IGNORECASE)
regex_going_to_vb = re.compile(r"(?:\s+|^)going/[^\s/]+\s+to/[^\s/]+\s+\S+/VB[DGNPZ]?(?:\s+|$)", re.IGNORECASE)

# Single comma
regex_single_comma = re.compile(r"^,/[^\s/]+$")

# Multi-character punctuation
regex_mc_punctuation = re.compile(r'^[!"#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~]{2,}/[^\s/]+$')

# Common nouns
regex_noun_common = re.compile(r'^\S+/NNS?$')

# Proper nouns
regex_noun_proper = re.compile(r'^\S+/NN(?:PS|P)?$')

# Multi-character punctuation, without PoS tags
regex_mc_punctuation_no_tag = re.compile(r'^[!"#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~]+$')

# PoS tags
regex_pos = re.compile(r"(\S+)/[^\s/]+", re.IGNORECASE)

regex_newline = re.compile(r"\n+")

# Adverbs
regex_adverbs = re.compile(r'^\S+/(?:RB[RS]?|RP)$')

# wh-words
regex_wh_words = re.compile(r'^\S+/(?:WP\$?|WRB|WDT)$')

# Words in uppercase (3 or more letters long)
regex_uppercase_words_3_chars = re.compile(r'^[A-Z]{3,}/[^\s/]+$')

# Slang acronyms
slang_acronyms_alternation = "(?:" + "|".join(slang_acronyms) + ")"
regex_slang_acronyms = re.compile(r'^{}/[^\s/]+$'.format(slang_acronyms_alternation), re.IGNORECASE)


def extract_features(comment):
    features = np.zeros((173,))

    extract_features_15_through_17(comment, features)

    comment = extract_feature_6(comment, features)

    tokens = regex_tokenizer.split(comment)

    for idx, token in enumerate(tokens):
        if not token or token == "\n":
            continue

        token = token.strip()

        extract_features_1_through_5(features, token)

        extract_features_7_through_10(features, token)

        extract_features_11_through_14(features, token)

    return features


def extract_features_7_through_10(features, token):
    # Feature 7: Number of commas
    match_single_comma = regex_single_comma.match(token)
    if match_single_comma:
        features[6] += 1

    # Feature 8: Number of multi-character punctuation tokens
    match_mc_punctuation = regex_mc_punctuation.match(token)
    if match_mc_punctuation:
        features[7] += 1

    # Feature 9: Number of common nouns
    match_noun_common = regex_noun_common.match(token)
    if match_noun_common:
        features[8] += 1

    # Feature 10: Number of proper nouns
    match_noun_proper = regex_noun_proper.match(token)
    if match_noun_proper:
        features[9] += 1


def extract_feature_6(comment, features):
    # Feature 6: Number of future tense verbs
    # It's better to extract feature 6 before tokenizing the comment down below...
    comment, count = regex_ftv.subn(" ", comment)
    features[5] += count

    comment, count = regex_gonna_vb.subn(" ", comment)
    features[5] += count

    comment, count = regex_going_to_vb.subn(" ", comment)
    features[5] += count

    return comment


def extract_features_11_through_14(features, token):
    # Feature 11: Number of adverbs
    if regex_adverbs.match(token):
        features[10] += 1

    # Feature 12: Number of wh-words
    if regex_wh_words.match(token):
        features[11] += 1

    # Feature 13: Slang acronyms
    if regex_slang_acronyms.match(token):
        features[12] += 1

    # Feature 14: Number of words in uppercase (>= 3 letters long)
    if regex_uppercase_words_3_chars.match(token):
        features[13] += 1


def extract_features_15_through_17(comment, features):
    # Feature 17: Number of sentences
    no_tag_comment = regex_pos.sub(r"\1", comment).strip()
    num_sentences = len(regex_newline.split(no_tag_comment))
    features[16] = num_sentences

    # Feature 15: Average length of sentences, in tokens
    no_tag_no_newline_comment = no_tag_comment.replace("\n", " ")
    tokens = list(filter(lambda x: x and x != "\n", regex_tokenizer.split(no_tag_no_newline_comment)))
    num_tokens = len(tokens)
    features[14] = num_tokens / num_sentences

    # Feature 16: Average length of tokens, excluding punctuation-only tokens, in characters
    tokens_exc_multicharacter_punct = filter(lambda x: not regex_mc_punctuation_no_tag.match(x), tokens)
    sum_tokens_length = functools.reduce(lambda acc, x: acc + len(x), tokens_exc_multicharacter_punct, 0)
    features[15] = sum_tokens_length / num_tokens


def extract_features_1_through_5(features, token):
    # Feature 1: Number of first-person pronouns
    match_fpp = regex_fpp.match(token)
    if match_fpp:
        features[0] += 1

    # Feature 2: Number of second-person pronouns
    match_spp = regex_spp.match(token)
    if match_spp:
        features[1] += 1

    # Feature 3: Number of third-person pronouns
    match_tpp = regex_tpp.match(token)
    if match_tpp:
        features[2] += 1

    # Feature 4: Number of coordinating conjunctions
    match_cc = regex_cc.match(token)
    if match_cc:
        features[3] += 1

    # Feature 5: Number of past tense verbs
    match_ptv = regex_ptv.match(token)
    if match_ptv:
        features[4] += 1


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    # This shouldn't be necessary, but for sanity...
    comment = comment.strip()

    if not comment:
        logging.warning("Ignoring empty comment...")
        return np.zeros((173,))

    features = extract_features(comment)

    return features


def encode_label(label):
    encoding = {
        "Left": 0,
        "Center": 1,
        "Right": 2,
        "Alt": 3
    }

    return encoding[label]


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    for i, datum in enumerate(data):
        body = datum['body']
        cat = datum['cat']
        encoded_cat = encode_label(cat)
        features = extract1(body)
        features = np.append(features, [encoded_cat], axis=0)
        feats[i] = features

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
