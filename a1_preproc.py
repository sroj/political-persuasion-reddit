import sys
import argparse
import os
import json

indir = 'data'


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
# indir = '/u/cs401/A1/data/';


def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        print('TODO')
    if 2 in steps:
        print('TODO')
    if 3 in steps:
        print('TODO')
    if 4 in steps:
        print('TODO')
    if 5 in steps:
        print('TODO')
    if 6 in steps:
        print('TODO')
    if 7 in steps:
        print('TODO')
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')

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
            max_lines = args.max

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

    if args.max > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
