import argparse
import html
import json
import os
import re
import sys
import spacy

# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# TODO Restore this!
# prefix_a1 = '/u/cs401/A1/'
prefix_a1 = ''
indir = prefix_a1 + 'data/';

# prefix_wordlist = '/u/cs401/'
prefix_wordlist = ''
wordlists_dir = prefix_wordlist + 'Wordlists/'

nlp = spacy.load('en', disable=['parser', 'ner'])


def remove_newlines(text):
    return text.replace("\n", " ").replace("\r", " ")


def remove_urls(text):
    return re.sub(r'(\bhttps?:?//[\w./\-:]+)|(\bwww\.[\w./\-:]+)', repl='', string=text)


def remove_html_char_codes(modComm):
    return html.unescape(modComm)


def read_all_abbreviations():
    files = ["abbrev.english", "pn_abbrev.english"]
    return set(read_files_by_line(wordlists_dir, files))


def read_proper_name_abbreviations():
    files = ["pn_abbrev.english"]
    return set(read_files_by_line(wordlists_dir, files))


def read_files_by_line(directory, files):
    lines = list()

    for file in files:
        with open(directory + file) as f:
            for line in f:
                lines.append(line.strip())

    return lines


def read_stopwords():
    files = ["StopWords"]

    words = set()

    for file in files:
        with open(wordlists_dir + file) as f:
            for line in f:
                words.add(line.strip())

    return words


def split_punctuation(modComm):
    abbreviations = read_all_abbreviations()

    abbreviations_regex = "(" + "|".join(abbreviations).replace(".", "\.") + ")"

    number_with_separator_regex = r"\d{1,3}(,\d{3})+(\.\d+)?"
    number_without_separator_regex = r"\b\d+\b"

    # TODO Maybe improve handling of cases like: ?:# i.e. multiple punctuation concatenated together
    pattern = abbreviations_regex + "|" \
              + number_with_separator_regex + "|" \
              + number_without_separator_regex + "|" \
              + r'([!"#$%&()*+,-./:;<=>?@\[\\\]^_{|}~]+)'

    result = re.sub(pattern=pattern, repl=repl_punctuation, string=modComm)

    # Remove repeated spaces
    result = re.sub(r' {2,}', repl=' ', string=result)

    return result


def repl_punctuation(matchobj):
    start = matchobj.start()
    end = matchobj.end()
    print("Start, end: {}, {}".format(start, end))
    match = matchobj.group(0)

    if match is not None:
        return " " + match + " "

    return match


def split_clitics(modComm):
    verb_tokens = "(ca|had|ai|am|are|could|dare|did|does|do|has|have|is|need|must|ought|should|was|were|wo|would)"

    # Handle "verb + not"
    modComm = re.sub(r"\b" + verb_tokens + r"n't\b", repl=r"\1 n't", string=modComm, flags=re.IGNORECASE)

    # Handle possessives and has/is
    modComm = re.sub(r"(\w+)'s", repl=r"\1 's", string=modComm, flags=re.IGNORECASE)

    # Handle plural possessives
    modComm = re.sub(r"(\w+)s'", repl=r"\1s '", string=modComm, flags=re.IGNORECASE)

    # Handle 've (e.g. should've)
    modComm = re.sub(r"(\w*[a-zA-Z])'ve\b", repl=r"\1 've", string=modComm, flags=re.IGNORECASE)

    # Handle 'm (e.g. I'm)
    modComm = re.sub(r"\b([Ii])'m\b", repl=r"\1 'm", string=modComm, flags=re.IGNORECASE)

    # Handle 're (e.g. You're)
    modComm = re.sub(r"(\w*[a-zA-Z])'re\b", repl=r"\1 're", string=modComm, flags=re.IGNORECASE)

    # Handle 'll (e.g. I'll)
    modComm = re.sub(r"(\w*[a-zA-Z])'ll\b", repl=r"\1 'll", string=modComm, flags=re.IGNORECASE)

    # Handle 'd (e.g. She'd)
    modComm = re.sub(r"(\w*[a-zA-Z])'d\b", repl=r"\1 'd", string=modComm, flags=re.IGNORECASE)

    return modComm


def remove_stopwords(modComm):
    stopwords = read_stopwords()

    stopwords_regex = [sw + r"(?:/[\S$]+)*" for sw in stopwords]

    regex = r"(?:\s|^)(?:" + "|".join(stopwords_regex) + r")(?=\s|$)"

    modComm = re.sub(regex, "", modComm, flags=re.IGNORECASE)

    return re.sub(r'\s{2,}', repl=' ', string=modComm)


def separate_sentences(modComm):
    pn_abb = read_proper_name_abbreviations()
    non_pn_abb = read_all_abbreviations() - pn_abb

    # The last dot on proper name abbreviation should be removed to make matching easier later
    # pn_abb = {re.sub(r"(\w+)\.$", r"\1", abb) for abb in pn_abb}

    pn_abb_regex = "(?P<pn_abb>(" + "|".join(pn_abb).replace(".", "\.") + r")\ +)"
    non_pn_abb_regex = "(?P<non_pn_abb>(" + "|".join(non_pn_abb).replace(".", "\.") + r")\ +)"

    modComm = re.sub(
        r'''
        ((?P<period>\."?)\ *(?=\w+))
        |
        {pn_abb}(?=\w+)
        |
        {non_pn_abb}(?=\w+)
        '''.format(pn_abb=pn_abb_regex, non_pn_abb=non_pn_abb_regex),
        repl_sentence,
        modComm,
        flags=re.VERBOSE
    )

    modComm = re.sub(r'([!?]+)\ *(?=[A-Z]+)', r"\1\n", modComm)

    return modComm


def repl_sentence(matchobj):
    pn_abb = matchobj.group("pn_abb")
    non_pn_abb = matchobj.group("non_pn_abb")
    period = matchobj.group("period")

    if pn_abb:
        return pn_abb
    elif non_pn_abb:
        next_character = matchobj.string[matchobj.end("non_pn_abb")]
        if next_character.isupper():
            return non_pn_abb + "\n"
        else:
            return non_pn_abb
    else:
        return period + "\n"


def tag_part_of_speech(modComm):
    tokens = nlp(modComm)

    tagged_comment = ""

    for token in tokens:
        tag = token.tag_
        token_length = len(token)
        idx = token.idx
        tagged_comment += " " + modComm[idx:idx + token_length] + "/" + tag

    return tagged_comment.strip()


def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = comment
    if 1 in steps:
        print('Removing newline characters')
        modComm = remove_newlines(modComm)
    if 2 in steps:
        print("Replacing HTML character codes")
        modComm = remove_html_char_codes(modComm)
    if 3 in steps:
        print('Removing urls')
        modComm = remove_urls(modComm)
    if 4 in steps:
        print('Splitting punctuation')
        modComm = split_punctuation(modComm)
    if 5 in steps:
        print('Splitting clitics')
        modComm = split_clitics(modComm)
    if 6 in steps:
        print('Tagging with part-of-speech')
        modComm = tag_part_of_speech(modComm)
    if 7 in steps:
        print('Removing stop words')
        modComm = remove_stopwords(modComm)
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('Adding newline after each sentence')
        modComm = separate_sentences(modComm)
    if 10 in steps:
        print('Lower-casing text')
        modComm = modComm.lower()

    return modComm


def sample_data(data, start_index, end_index):
    """
    Takes a slice of data, wrapping around if needed

    :param data: data to be sampled
    :param start_index: start index
    :param end_index: end index, can be less than start index, in which case the slicing operation wraps around the end
    :return: a slice of data starting at start_index and endind at end_index, wrapping around if necessary
    """
    if end_index >= start_index:
        print("Slicing normally")
        return data[start_index:end_index]
    else:
        print("Slicing circularly")
        return data[start_index:] + data[:end_index]


def remove_unused_fields(data, keys_to_keep):
    filtered_data = list()
    for datum in data:
        filtered_comment = dict()
        for key in keys_to_keep:
            if key in datum:
                filtered_comment[key] = datum[key]
        filtered_data.append(filtered_comment)

    return filtered_data


def label_data(data, label):
    for datum in data:
        datum["cat"] = label


def preprocess_bodies(data):
    preprocessed_data = list()
    key_body = "body"

    for datum in data:
        if key_body in datum:
            datum[key_body] = preproc1(datum[key_body])
            preprocessed_data.append(datum)
        else:
            print("WARNING Found post without a body")

    return preprocessed_data


def main(args):
    student_id = args.ID[0]
    print("Student ID is {}".format(student_id))

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            num_comments = len(data)

            print("File {} has {} comments".format(fullFile, num_comments))

            # TODO: select appropriate args.max lines
            max_lines = int(args.max)

            start_index = student_id % num_comments
            end_index = (start_index + max_lines) % num_comments

            print("Sampling {} comments starting at {} and ending at {}".format(max_lines, start_index, end_index))
            sampled_data = sample_data(data, start_index, end_index)
            print("The sampled dataset contains {} comments".format(len(sampled_data)))

            # TODO: read those lines with something like `j = json.loads(line)`
            sampled_data = [json.loads(line) for line in sampled_data]

            # TODO: choose to retain fields from those lines that are relevant to you
            keys_to_keep = [
                "id",
                "score",
                "controversiality",
                "subreddit",
                "author",
                "body",
                "ups",
                "downs"
            ]
            sampled_data = remove_unused_fields(sampled_data, keys_to_keep)

            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            label_data(sampled_data, file)

            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            preprocess_bodies(sampled_data)

            # TODO: append the result to 'allOutput'
            allOutput.append(sampled_data)

            print("----------\n")

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if int(args.max) > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
