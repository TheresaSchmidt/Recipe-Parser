"""
Theresa Schmidt, Matrikelnummer 2565903

Takes ParZu and brat annotations files for a recipe and aligns them into
ConLL2003 format: TOKEN, POS-TAG, CHUNK-TAG, (COOK) LABEL
The chunk-tag is always = for now
"""


from collections import defaultdict
import numpy as np
import pandas as pd
import argparse
import pickle #might not be necessary in the end
import nltk


def flatten_data(data):
    tokens = []
    labels = []
    for index,row in data.iterrows():
        start,label,chunk = row
        t,l = flatten_chunk(label,chunk)
        tokens.extend(t)
        labels.extend(l)
    return list(zip(tokens,labels))

def flatten_chunk(label, chunk):
    #tokens = chunk.replace("/"," / ").replace(":"," :").replace(u"\u00b0C",u"\u00b0 C").split()
    #tokens = nltk.word_tokenize(chunk,language="german")
    tokens = chunk
    # think about putting all different codings for dash, hyphen, etc. into one regex and using it instead of "-";
    # degree-celsius-character
    labels = []
    t_len=len(tokens)
    if t_len == 1:
        labels = ["U-"+label]
    elif t_len == 2:
        labels = ["B-"+label, "L-"+label]
    elif t_len > 2:
        labels = ["B-"+label] + (t_len - 2)*["I-"+label] + ["L-"+label]
    else:
        raise RuntimeError ("Unexpected Error: parsed chunk is empty")
    return (tokens,labels)

def execute(brat_file, parzu_file, out_file, debug):
    with open(parzu_file, "r")as parses:
        """while line != "":
            tokens = [] #tokenized sentence
            pos = [] #pos tags for all tokens
            while line != "\n": #sentence by sentence
                es = line.split("\t")
                tokens.append(es[0])
                pos.append(es[3])"""
                
        with open(brat_file, "r") as anns:
            doc_list = []
            for line in anns:
                if (line != "\n"): #and (line != ""): #recipe by recipe ??sinnvoll / wichtig? Idee: Tipps leichter ueberspringen; stimmt das?
                    # read in annotations for role labels
                    if line.startswith("T"):
                        line = line.split()
                        label = line[1]
                        start = line[2]
                        chunk = line[4:]
                        if "bzw" in chunk:
                            chunk = " ".join(chunk).replace("bzw","bzw.").split()
                        #print(chunk)
                        #ref, ann, chunk = line.split("\t")
                        #label, start, _ = ann.split()
                        doc_list.append([int(start), label, chunk])
            #else: #unfortunately "for line in anns" does not capture the EOF line ""
            data = pd.DataFrame(doc_list)
            data = data.sort_values(by=[0])
            annotation = flatten_data(data)
            # go through labels and align them with the text
            with open(out_file, "a") as f:
                annotation.append((None,None)) #for the final loop
                t,l =annotation.pop(0)
                while annotation:
                    p_line = parses.readline().split("\t") # conll: token, _, _, pos, ...
                    #f.write(t + l + str(p_line) + "\n")
                    if p_line == [""]: #this should never happen as we use the same tokenizer for annotated and tagged file
                        #but it does happen for "bzw."
                        raise RuntimeError("end of recipe; couldn't find token '" + t +"' in file " + parzu_file)
                    if p_line == ["\n"]:
                        f.write("\n")
                        #print(annotation)
                    elif p_line[1] == t:
                        # found labelled token
                        f.write(p_line[1] + "\t" + p_line[4] + "\tO\t" + l + "\n")
                        t,l = annotation.pop(0)
                        #f.write("accept\n")
                        #print(p_line[1] + "\t" + p_line[4] + "\tO\t" + l)
                    else:
                        # found O labelled token
                        if debug:
                            f.write("*" + p_line[1] + "\t" + t + "\t" + p_line[4] + "\tO\t" + l + "\n")
                        f.write(p_line[1] + "\t" + p_line[4] + "\tO\tO\n")
                        #print("O")
                        #f.write("O\n")
                #print rest of input as O labelled
                p_line= parses.readline().split("\t")
                while p_line != [""]:
                    if p_line == ["\n"]:
                        f.write("\n")
                    else:
                        f.write(p_line[1] + "\t" + p_line[4] + "\tO\tO\n")
                    p_line=parses.readline().split("\t")
            #??start new recipe
            doc_list=[]
                    
            #start new sentence


if __name__ == "__main__":

    # parser for command line arguments
    parser = argparse.ArgumentParser(
        description = """Combine ParZu output with brat annotation. Output format: txt file with
                        ConLL2003 format (columns: TOKEN, POS-TAG, CHUNK-TAG, (COOK) LABEL)""")
    parser.add_argument("ann", metavar="brat_file", #TODO: only single recipe files
                        help="""Path to a brat annotated file for a single recipe. If
                        no parzu and output file are specified, this file and
                        the annotation file should be in the same folder.""")
    parser.add_argument("-p", "--parzu", dest="parzu", metavar="parzu_file",
                        help="""Path to a corresponding ParZu output file (format:
                                CoNLL with columns index token _ _ POS-tag _*).
                                Default: prefix from ann-file + '.txt.parzu'""")
    parser.add_argument("-o", "--output-file", dest="out", metavar="output_file",
                        help="""Specify an outputfile. Default: prefix from annotation
                                file + '.txt'""")
    parser.add_argument("-d", "--debug", dest="debug", const=True, default=False, nargs="?",
                        help="""""")
    args = parser.parse_args()
    
    # Determine file names
    if args.parzu == None:
        args.parzu = str(args.ann)[:-3] + "txt.parzu"
    if args.out == None:
        args.out = str(args.ann)[:-3] + "txt"

    # Start execution
    print("zipping "  + args.ann + " and " + args.parzu + " into " + args.out)
    execute(args.ann, args.parzu, args.out, args.debug)
    
