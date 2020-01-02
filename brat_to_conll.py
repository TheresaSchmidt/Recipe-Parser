"""
Theresa Schmidt, Matrikelnummer 2565903

Takes ParZu and brat annotations files for a recipe and aligns them into
ConLL2003 format: TOKEN, POS-TAG, CHUNK-TAG, (COOK) LABEL
The chunk-tag is always = for now

Python 3.6.9
"""


from collections import defaultdict
from collections import OrderedDict
import numpy as np
import pandas as pd
import argparse
import pickle #might not be necessary in the end
import nltk


def flatten_data(data):
    tokens = []
    labels = []
    references = []
    for index,row in data.iterrows():
        start,label,chunk,ref = row
        t,l = flatten_chunk(label,chunk)
        tokens.extend(t)
        labels.extend(l)
        references.extend([ref]*len(t))
    return list(zip(tokens,labels,references))

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

#def execute(brat_file, parzu_file, out_file, debug):
    
def read_annotation (brat_file):
    with open(brat_file, "r") as anns:
        _labels = []
        events = dict()
        relations = dict()
        aliasses = []
        for line in anns:
            if (line != "\n"): #and (line != ""): #recipe by recipe ??sinnvoll / wichtig? Idee: Tipps leichter ueberspringen; stimmt das?
                # read in annotations for role labels
                line = line.split()
                if line[0].startswith("T"):
                    ref = line[0] #new; mitliefern
                    label = line[1]
                    start = line[2]
                    chunk = line[4:]
                    if "bzw" in chunk:
                        chunk = " ".join(chunk).replace("bzw","bzw.").split() #do I still need this?
                    _labels.append([int(start), label, chunk, ref])
                elif line[0].startswith("E"):
                    # line = [ref] + [deprel, ':', target_ref]*n (n>=2)
                    ref = line[0]
                    """ if annotation file is not tokenized
                    head = line[1].split(":")[1] # prior: line[1] = deprel:ref
                    children = line[2:] #format: 'deprel:ref'
                    """
                    head = line[3]
                    children = [(line[i],line[i+2]) for i in range(4,len(line),3)]
                    events[ref]=(head,children)
                elif line[0].startswith("*"): 
                    deprel = line[1]
                    if deprel != "Alias":
                        raise RuntimeError ("Unexpected error: transitive relation is ", deprel, " , not 'Alias'.")
                    args = line [2:]
                    aliasses.append(args) # copy attributes in both directions
                elif line[0].startswith("R"):
                    # line = [ref, deprel] + ['Arg1', ':', ref] + ['Arg2', ':', ref]
                    ref = line[0]
                    deprel = line[1]
                    """ if annotation file is not tokenized
                    head = line[2].split(":")[1]
                    child = line[3].split(":")[1]
                    """
                    head = line[4]
                    child = line[7]
                    relations[ref]=(deprel,head,child)
        #else: #unfortunately "for line in anns" does not capture the EOF line ""
        data = pd.DataFrame(_labels)
        data = data.sort_values(by=[0])
        # annotation = flatten_data(data)
        return (flatten_data(data),events,relations,aliasses)
                                            
def align_parzu(annotation, parzu_file, out_file, debug):
    with open(parzu_file, "r")as parses:
        # go through labels and align them with the text
        with open(out_file, "a") as f:
            annotation.append((None,None,None)) #for the final loop
            t,l,r = annotation.pop(0)
            token_index = 0
            conll = OrderedDict()
            p_cache = []
            ann_cache = []
            #print(t,l,r)
            #print(annotation)
            #print(p_cache)
            #print(token_index)
            while annotation:
                if p_cache == []:
                    token_index += 1
                    p_line = parses.readline().split("\t") # conll: token, _, _, pos, ...                        
                else:
                    t,l,r = (None,None,None)
                    token_index, p_line = p_cache.pop(0)
                    if p_cache == []:
                        t,l,r = annotation.pop(0)
                if p_line == [""]: #this should never happen as we use the same tokenizer for annotated and tagged file
                    #but it does happen with sloppy annotation, e.g. for "bzw."
                    #these mistakes have to be corrected by hand in the annotation file (or parzu file)
                    phrase = t
                    if l[0]=="B":
                        for t,l,r in annotation:
                            phrase += " " + t
                            if l[0]=="L":
                                break
                    raise RuntimeError("end of recipe; couldn't find phrase '" + phrase +"' in file " + parzu_file)
                if p_line == ["\n"]:
                    token_index -= 1
                    conll["S"+str(token_index)] = "\n" #sentence boundaries have to be unique so as not to override each other
                elif p_line[1] == t:
                    # check chunk
                    if l[0]=="B":
                        conll_cache = OrderedDict()
                        conll_cache[r] = [[token_index, p_line[1], p_line[4], l]] #id, token, (u)pos, xpos
                        ann_cache.append((t,l,r))
                        p_cache = [(token_index,p_line)]
                        while True: #l[0]!="L":
                            t,l,r = annotation.pop(0)
                            ann_cache.append((t,l,r))
                            token_index += 1
                            p_line = parses.readline().split("\t") # conll: token, _, _, pos, ...
                            p_cache.append((token_index,p_line))
                            if p_line == ["\n"]:
                                token_index -= 1
                                annotation = [(t,l,r)] + annotation
                                # if there is a sentence boundary inside a tag sequence it should be deleted
                                # X is only needed to measure how often this occurred
                                conll_cache["X"+str(token_index)] = " "
                            elif p_line[1] == t:
                                conll_cache[r].append([token_index, p_line[1], p_line[4], l]) #id, token, (u)pos, xpos
                                if l[0]=="L":
                                    ann_cache = []
                                    #print (p_cache)
                                    p_cache = []
                                    for ref,entry in conll_cache.items():
                                        conll[ref] = entry
                                    break
                                    #print(t,l,r)
                                    #print(annotation)
                                    #print(p_cache)
                                    #print(token_index)
                            else:
                                annotation = [(None,None,None)] + ann_cache + annotation
                                ann_cache = []
                                break
                        t,l,r = annotation.pop(0)
                        
                    elif l[0]=="U":
                        conll [r] = [[token_index, p_line[1], p_line[4], l]] #id, token, (u)pos, xpos(ner label)
                        t,l,r = annotation.pop(0)
                     #write and cancel or don't write and release
                else:
                    # found O labelled token
                    #line without tag and therefore without reference
                    if debug:
                        f.write("*" + p_line[1] + "\t" + str(t) + "\t" + p_line[4] + "\tO\t" + str(l) + "\n")
                    conll["N"+str(token_index)] = [token_index, p_line[1], p_line[4] + "O"] #id, token, (u)pos, xpos(ner label)

            #rest of input has label O
            p_line= parses.readline().split("\t")
            while p_line != [""]:
                if p_line == ["\n"]:
                    token_index -= 1
                    conll["S"+str(token_index)] = "\n"
                else:
                    conll["N"+str(token_index)] = [token_index, p_line[1], p_line[4] + "O"] #id, token, (u)pos, xpos(ner label)
                p_line=parses.readline().split("\t")
                token_index += 1
            return conll

def write_conll2003(conll, out_file):
    with open(out_file, "a") as f:
        for ref,entry in conll.items():
            if ref[0]=="X":
                pass
            elif ref[0]=="S":
                f.write("\n")
            elif ref[0]=="T":
                for e in entry:
                    f.write(e[1] + "\t" + e[2] + "\tO\t" + e[3] + "\n") #token,pos,chunk,label
            elif ref[0]=="N":
                f.write(entry[1] + "\t" + entry[2] + "\tO\tO\n") #token,pos,chunk,label
            else:
                raise RuntimeError ("Unexpected reference " + ref)
            #f.write(ref+str(entry)+"\n") #TODO: \t
        f.write("\n")

def write_conllu(conll,out_file):
    with open(out_file, "a") as f:
        for ref,entry in conll.items():
            if ref[0]=="X" or ref[0]=="S":
                pass
            elif ref[0]=="T":
                # id,form,lemma,(u)pos,xpos(label),feats,head,deprel,deps,misc
                for line in conll[ref]:
                    if len(line) > 5: # token is child of several heads
                        line = line[:4] + list(set(line[4:]))
                        (head,deprel) = line[4]
                        if len(line) > 5:
                            f.write(str(line[0]) + "\t" + line[1] + "\t_\t" + line[2]
                                + "\t" + line[3] + "\t_\t" + head + "\t" + deprel + "\t" + str(list(set(line[5:]))) + "\t_\n")
                        else:
                            f.write(str(line[0]) + "\t" + line[1] + "\t_\t" + line[2]
                                + "\t" + line[3] + "\t_\t" + head + "\t" + deprel + "\t_\t_\n")
                    elif len(line) > 4: # token has one head
                        (head,deprel) = line[4]
                        f.write(str(line[0]) + "\t" + line[1] + "\t_\t" + line[2]
                                + "\t" + line[3] + "\t_\t" + head + "\t" + deprel + "\t_\t_\n")
                    else: # token has no head so it must be root (requirement of the conllu format)
                        f.write(str(line[0]) + "\t" + line[1] + "\t_\t" + line[2] + "\tO\t_\t0\troot\t_\t_\n")
                #f.write(entry[1] + "\t" + entry[2] + "\tO\t" + entry[3] + "\n") 
            elif ref[0]=="N":
                line = conll[ref]
                # id,form,lemma,(u)pos,xpos(label),feats,head,deprel,deps,misc
                f.write(str(line[0]) + "\t" + line[1] + "\t_\t" + line[2] + "\tO\t_\t0\troot\t_\t_\n")
                #f.write(entry[1] + "\t" + entry[2] + "\tO\tO\n")
            else:
                raise RuntimeError ("Unexpected reference " + ref)
        f.write("\n")

def add_dependencies(conll,events,relations,aliasses):
    for event in events:
        head_ref,children = events[event]
        # If a key error occurs here, check the corresponding
        # annotation file for a line that doesn't start with one of {T,E,R,#,*}
        # This will be the case for chunks that contain the newline character.
        head = conll[head_ref][0][0]
        for deprel,ref in children:
            """ if annotation file is not tokenized
            [deprel,ref] = c.split(":") # prior: c=deprel:ref
            """
            # delete trailing integers in dependency names
            try:
                i = int(deprel[-1])
                deprel = deprel[:-1]
            except:
                ValueError 
            if ref[0]=="T":
                for e in conll[ref]:
                    e.extend([(str(head),deprel)])
            elif ref[0]=="E":
                _ref,_ = events[ref]
                for e in conll[_ref]:
                    e.extend([(str(head),deprel)])
            else:
                raise RuntimeError ("Unexpected reference ",ref," as child of event ",event)

    for relation in relations:
        deprel,head_ref,child = relations[relation]
        # delete trailing integers in dependency names
        try:
            i = int(deprel[-1])
            deprel = deprel[:-1]
        except:
            ValueError 
        # get head index; TODO: index of first token of head chunk or 'i-j' notation?
        if head_ref[0]=="T":
            head=conll[head_ref][0][0]
        elif head_ref[0]=="E":
            head_ref,_ = events[head_ref]
            head=conll[head_ref][0][0]
        # add head to tokens of child
        if child[0]=="T":
            for c in conll[child]:
                c.extend([(str(head),deprel)])
        elif child[0]=="E":
            _ref,_ = events[child]
            for c in conll[_ref]:
                c.extend([(str(head),deprel)])
        else:
            raise RuntimeError ("Unexpected reference ",child," as child of relation ",relation)
    
    for a in aliasses:
        heads = [] #head, deprel, head, ..., deprel
        children = []
        for alias in a:
            # find all heads with relations (unfortunatly, recipes turn out to be graphs, not trees)
            heads.extend(conll[alias][0][4:])
            # find all children
            for c in conll:
                if c[0]=="T":
                    if alias in conll[c][0][4:]:
                        children.append(c)
        print("Number of heads of aliasses ",str(a)," is: ",len(heads)/2)
        for alias in a:
            # add all heads with relations
            for c in conll[alias]:
                c.extend(heads) #doubles will be cleaned out later; do not matter in tree projection anyway
        # add aliasses in children
        for ref in children:
            agenda = conll[ref][0][4:].copy()
            while agenda:
                head = agenda.pop(0)
                deprel = agenda.pop(0)                
                for alias in a:
                    if head == alias:
                        syns = a.copy()
                        syns.remove(alias)
                        for s in syns:
                            for c in conll[ref]:
                                c.extend([(str(s),deprel)])
    
        
    #clean conll from doubles        
    #fill heads with deprel root and head 0 ??
    return conll
                                            


if __name__ == "__main__":

    # parser for command line arguments
    arg_parser = argparse.ArgumentParser(
        description = """Combine ParZu output with brat annotation. Currently, the annotation file has
                        to be tokenized prior to execution. Output formats: txt file with CoNLL2003
                        format (realized columns: TOKEN, POS-TAG, _ , (COOK) LABEL) and conllu file
                        with CoNLL-U format(relized columns: ID, TOKEN, _ , POS-TAG, (COOK) LABEL, _,
                        HEAD, DEPREL, _ , _).""")
    arg_parser.add_argument("ann", metavar="brat_file", #TODO: only single recipe files
                        help="""Path to a brat annotated file for a single recipe. If
                        no parzu and output file are specified, this file and
                        the annotation file should be in the same folder.""")
    arg_parser.add_argument("-p", "--parzu", dest="parzu", metavar="parzu_file",
                        help="""Path to a corresponding ParZu output file (format:
                                CoNLL with columns index token _ _ POS-tag _*).
                                Default: prefix from ann-file + '.txt.parzu'""")
    arg_parser.add_argument("-c3", "--conll2003", dest="conll2003",
                            metavar="conll2003_output_file",
                        help="""Specify an outputfile for the ConLL2003 format.
                                Default: prefix from annotation file + '.conll03'""")
    arg_parser.add_argument("-cu", "--conllu", dest="conllu", metavar="conllu_output_file",
                        help="""Specify an outputfile for the ConLL-U format. Default:
                                prefix from annotation file + '.conllu'""")
    arg_parser.add_argument("-o", "--output_prefix", dest="out", metavar="output_prefix",
                        help="""Specify a prefix to be used in the output files'
                                names. If file names are specified by -c3 and -cu, -o
                                has no effect. Default: prefix from annotation file. """)
    arg_parser.add_argument("-t", "--only_tags", dest="tags", const = True, default = False,
                            action = 'store_const',
                        help="""Use this flag to disable reading in the dependencies
                                from the annotation.""")
    arg_parser.add_argument("-d", "--only_dependencies", dest="dependencies", const = True,
                            default = False, action = 'store_const',
                        help="""Use this flag to output only a CoNLL-U file and no
                                CoNLL2003 file.""")
    """arg_parser.add_argument("-d", "--debug", dest="debug", const=True, default=False, nargs="?",
                        help="""""") #TODO reform debug """
    args = arg_parser.parse_args()

    
    # Determine file names
    # TODO: integrate defaults directly in the arguments
    args.debug = False
    if args.parzu == None:
        args.parzu = str(args.ann)[:-3] + "txt.parzu"
    if args.conll2003 == None:
        if args.out:
            args.conll2003 = str(args.out) + ".conll03"
        else:
            args.conll2003 = str(args.ann)[:-3] + "conll03"
    if args.conllu == None:
        if args.out:
            args.conll2003 = str(args.out) + ".conllu"
        else:
            args.conllu = str(args.ann)[:-3] + "conllu"

    # Start execution
    print("zipping "  + args.ann + " and " + args.parzu + " into " + args.conll2003 + " with labels and into " + args.conllu + " with labels and dependecies")
    annotation,events,relations,aliasses = read_annotation(args.ann)
    conll = align_parzu(annotation, args.parzu, args.conll2003, args.debug)
    if not args.dependencies:
        write_conll2003(conll,args.conll2003)
    if not args.tags:
        conll = add_dependencies(conll,events,relations,aliasses)
        write_conllu(conll,args.conllu)
    
