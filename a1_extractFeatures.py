import numpy as np
import sys
import argparse
import os
import json
import re

prefix_wordlist = ''
wordlists_dir = prefix_wordlist + 'Wordlists/'

regex_tokenizer = re.compile(r"[^\S\n]*(\n)+[^\S\n]*|[^\S\n]+")


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


first_person_pronouns = read_first_person_pronouns()
second_person_pronouns = read_second_person_pronouns()
third_person_pronouns = read_third_person_pronouns()

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


def extract_features(tokens):
    features = np.zeros((173,))

    for idx, token in enumerate(tokens):
        if not token:
            continue

        token = token.strip()

        # Feature 1: Number of first-person pronouns
        match = regex_fpp.match(token)
        if match:
            features[0] += 1

        # Feature 2: Number of second-person pronouns
        match = regex_spp.match(token)
        if match:
            features[1] += 1

        # Feature 3: Number of third-person pronouns
        match = regex_tpp.match(token)
        if match:
            features[2] += 1

        # Feature 4: Number of coordinating conjunctions
        match = regex_cc.match(token)
        if match:
            features[3] += 1

        # Feature 5: Number of past tense verbs
        match = regex_ptv.match(token)
        if match:
            features[4] += 1

    return features


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    # This shouldn't be necessary, but for sanity...
    comment = comment.strip()

    tokens = regex_tokenizer.split(comment)

    features = extract_features(tokens)

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
