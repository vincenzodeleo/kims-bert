# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import pkuseg
import numpy as np
from tokenizers import BertWordPieceTokenizer
from functools import reduce
import pandas as pd

def my_sample(x, k):
    try:
        return x.sample(k)
    except:
        return x

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False, use_chinese_characters = True, predicate_label=None, vocab_file = None):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        ## TODO: understand how to use the ontology in english to improve the tokenizer as it is done here
#        self.tokenizer_ch = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.lookup_table = self._create_lookup_table(use_chinese_characters, predicate_label)
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG


    def _create_lookup_table(self, use_chinese_characters=True, predicate_label=None):
        lookup_table = {}

        # predicate_label = 'name'        

        ### Creates a lookup table where the keys are the triples subjects and values are 
        # (i) the concatenation of predicate and object, (in case a complete spo file for the knowledge graph is used with subject - predicate - object)
        # (ii) the object in case of subject - object with no predicate (in case the spo file has just these two elements for each line)
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))

            # with open(spo_path, 'r', encoding='utf-8') as f:
                # for line in f:

            ### Filter triples based on different criteria
            df = pd.read_csv(spo_path, sep='\t', header = None)
            df.columns = ['SUBJECT','PREDICATE','OBJECT']        
            df = df[df.PREDICATE!="Link from a Wikipage to another Wikipage"] # this predicate is removed because its rate is too high and it si not useful
            
            if predicate_label is None:
                df_filtered = df
                #df_filtered = df.groupby(['SUBJECT','PREDICATE']).apply(lambda x: x.sample(1)).reset_index(drop=True).groupby(['SUBJECT']).apply(lambda x: my_sample(x,config.MAX_ENTITIES)).reset_index(drop=True)
            else:
                df_filtered = df[df['PREDICATE']==predicate_label].groupby(['SUBJECT']).apply(lambda x: x.sample(1)).reset_index(drop=True)
            for ind in df_filtered.index:                

                        # subj, pred, obje = line.strip().split("\t")
                    subj, pred, obje = df_filtered['SUBJECT'][ind], df_filtered['PREDICATE'][ind], df_filtered['OBJECT'][ind]

                    if self.predicate:
                        try:
                            if use_chinese_characters:
                                value = pred + obje
                            else:
                                value = pred + " " + obje
                        except:
                            print("[KnowledgeGraph] Bad spo:", ind)
                            continue

                    else:
                        value = obje
                    
                    value = " ".join(list(self.tokenizer.encode(value).tokens)[1:-1]) ## we need to insert the tokenized version of the triple text

                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        if predicate_label is not None:
            df_filtered.to_csv("./test/selected_triples__{}.spo".format(predicate_label))
        return lookup_table

    def search_element_in_list(self, offset_list, value):
        for i in range(len(offset_list)):
            if offset_list[i] == value:
                return True, i
        return False, -1

    def search_inside_element_in_list(self, list, tpl, pos):
        for i in range(len(list)):
            if list[i][pos] == tpl[pos]:
                return True,i
        return False,-1

    def add_knowledge_with_vm(self, sent_batch, df_mapped_sel, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """

        debug = False
        # debug = True


        if debug:
            print("\n#######################################################")
            print(sent_batch)
            print("Searching entities on DBPedia Spotlight...")

        entity_pos = []
        for i in df_mapped_sel[['uri','entity_data']].values:
            entity_uri = i[0]
            entity_text = i[1]['surface_form']
            entity_start_char = i[1]['surface_char_pos'][0] 
            entity_end_char = i[1]['surface_char_pos'][1] 
            if debug:
                print(entity_uri, entity_text, entity_start_char, entity_end_char)
            entity_pos.append((entity_start_char, entity_end_char, entity_uri))
        if debug:
            print("entity_pos:",entity_pos)
            print("Done Searching entities on DBPedia Spotlight")

        entity_pos.sort() ## We need to sort positions with respect to first word in the tuple ascending

        #split_sent_batch_ch = [self.tokenizer_ch.cut(sent) for sent in sent_batch]
        split_sent_batch = [self.tokenizer.encode(sent).tokens for sent in sent_batch]
        #split_sent_batch = [list(self.tokenizer.encode(sent[5:]).tokens)[0:-1] for sent in sent_batch]
        
        #split_sent_batch =[sent if len(sent) <= max_length else sent[0:max_length] for sent in split_sent_batch_base]

        if debug: 
            print(split_sent_batch)

        split_sent_batch_offsets = [list(self.tokenizer.encode(sent).offsets) for sent in sent_batch]
        #split_sent_batch_offsets = [list(self.tokenizer.encode(sent[5:]).offsets)[0:-1] for sent in sent_batch]

        if debug: 
            print(split_sent_batch_offsets)    

        # split_sent_batch_merged = []
        split_sent_batch_merged = split_sent_batch.copy()
        uri_vect = [''] * len(split_sent_batch_merged[0])
        num_removed_word = 0
        if entity_pos!=[]:            
            for ep in entity_pos:
                # if ep[2]=='http://dbpedia.org/resource/Sudan':
                #     print("Eccolo!")
                #     print(split_sent_batch_merged)
                for ssbo, ssb in zip(split_sent_batch_offsets, split_sent_batch_merged):
                    if debug:
                        print(ssb)
                    one_word_entity_found, list_pos = self.search_element_in_list(ssbo, ep[:2])
                    # nothing to do for entity defined by one single word
                    if debug:
                        print(len(uri_vect), list_pos, ep[2])
                    
                    if one_word_entity_found:
                        uri_vect[list_pos - num_removed_word] = ep[2]                    
                    else:
                        multiple_words_entity_found_start_pos,list_start_pos = self.search_inside_element_in_list(ssbo, ep[:2], pos=0)
                        multiple_words_entity_found_end_pos,list_end_pos = self.search_inside_element_in_list(ssbo, ep[:2], pos=1)
                        list_start_pos = list_start_pos - num_removed_word
                        list_end_pos = list_end_pos - num_removed_word
                        if multiple_words_entity_found_start_pos and multiple_words_entity_found_end_pos:
                            # using reduce() + lambda + list slicing
                            # merging list elements
                            # see: https://www.geeksforgeeks.org/python-merge-list-elements/
                            ssb[list_start_pos : list_end_pos+1] = [reduce(lambda i, j: i + ' ' + j, ssb[list_start_pos : list_end_pos+1])]
                            uri_vect[list_start_pos : list_end_pos+1] = [reduce(lambda i, j: i + ' ' + j, uri_vect[list_start_pos : list_end_pos+1])]
                            uri_vect[list_start_pos] = ep[2]
                            # TODO: add raise exception when we miss starting or ending position
                            
                            # We have to keep track of the number of the words that we are removing trom the array
                            # in order to allineate the entity position in the vector with the merged words
                            num_removed_word = num_removed_word + (list_end_pos-list_start_pos) 
                            #print("ciao")
                        else:
                            print("Error. Looking for %s. Surface form start position found: %s, end position found: %s" % (ep, multiple_words_entity_found_start_pos,multiple_words_entity_found_end_pos))
                            raise Exception


                    # split_sent_batch_merged = [ssb]
        # else:
        #     split_sent_batch_merged = split_sent_batch.copy()

        if debug:
            print("*********************************************")
            print("lenght split_sent_batch_merged", len(split_sent_batch_merged[0]))
            print(split_sent_batch_merged)

            print(uri_vect)
            print("*********************************************")

        # for sent in sent_batch:
        #     print(dir(self.tokenizer.encode(sent[5:])))
        #     break

        # import sys
        # sys.exit()

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        num_triples_added_batch = []
        
        for split_sent in split_sent_batch_merged:
        #for split_sent in split_sent_batch:

            # create tree - (Sentence Tree in the paper)
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []

            num_triples_added = []

            ## original sentence ""[CLS]" "Tim" Cook is visiting Beijing now"
            ## tokenized sentence (split_sent): ["[CLS]". "Tim", Cook, is, visiting, Beijing, now ]
            #print(split_sent)
            for token_pos, token in enumerate(split_sent):

                # if token=='claw':
                #     print("eccolo!")

                ## Each step of the for loop takes into account a single token:
                # "[CLS]"
                # "Tim"
                # Cook
                # is
                # ...

                ######################### IMPORTANT NOTICE #######################
                ### the following comments describe how this algorithm should work if:
                ###   - we consider a tokenizer that extracts only words and not named entities (e.g., "Tim" and "Cook" insteasd of "Tim Cook")
                ###   - the code splits each token in characters instead of words because of how the chinese written language works (a group of character is considered a unique concept based on their reciprocal proximity and there is no separation of words with spaces)


                ### Entity linking where each token from the phrase is looked up in among the KG's subjects
                ### The lookup uses the prebuilt lookup tabel.

                ### Lookup table
                ### SubjectX -> [pred1+obj1, pred2+obj2, ..., predN+objN] 
                ### N can be very big
                ### the default value for max_entities is 2

                #entities = list(self.lookup_table.get(token, []))[:max_entities]  # a subject can be associated with a lot of pred+object, so we have to fix a maximum number of those for us to take
                entities = list(self.lookup_table.get(uri_vect[token_pos], []))[:max_entities]  # a subject can be associated with a lot of pred+object, so we have to fix a maximum number of those for us to take                
                #print(entities)
                num_triples_added.append(len(entities))
                ### entities contains a list of pred+object associated with the subject

                ### TODO: can we improve the entity linking process using something more powerful than the string/token matching? 
                ### using string matching we can have problems like
                ### - ambiguity if a word has more than one meaning in the KG
                ### - you must use all the labels associated with an entity in the KG to build the SPO file 

                ### TODO: it is very important how you produce the KG spo file from your original KG. We can study different strategies to op"tim"ize this process.

                ### Adds the KG triples that can be "attached" to the token (word in english perhaps charachter in chinese)
                sent_tree.append((token, entities))

                ## first step with token: "[CLS]"
                ### entities = []
                ### sent_tree = [("[CLS]",[])]

                ## second step with token: "Tim"
                ### entities = []
                ### sent_tree = [("[CLS]",[]), ("Tim",[])]

                ## third step with token: "Cook"
                ### entities = ["CEOApple"] ## see how the lookup table is builf with a concatenation of strings TODO: check if it is applicable in english
                ### sent_tree = [("[CLS]",[]), ("Tim",[]), ("Cook",["CEOApple"])]

                
                ## fourth step with token: "is"
                ### entities = [] 
                ### sent_tree = [("[CLS]",[]), ("Tim",[]), ("Cook",["CEOApple"]), ("is",[])]


                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    
                    token_words = token.split(" ")     
                    #token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)] ## in english len(token) would be the length of the word, in chinese ????
                    token_pos_idx = [pos_idx+i for i in range(1, len(token_words)+1)] ## in english len(token) would be the length of the word, in chinese ????
                    #token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token_words)+1)]
                abs_idx = token_abs_idx[-1] ## we point abs_idx to the last position of the token we are examining

                ## first step with token: "[CLS]"
                ### token_pos_idx = [0]
                ### token_abs_idx = [0]
                ### abs_idx = 0
                
                ## second step with token: "Tim"
                ### token_pos_idx = [1, 2, 3]
                ### token_abs_idx = [1, 2, 3]
                ### abs_idx = 3

                ## third step with token: "Cook"
                ### token_pos_idx = [4, 5, 6, 7]
                ### token_abs_idx = [4, 5, 6, 7]
                ### abs_idx = 7

                ## fourth step with token: "is"
                ### token_pos_idx = [8, 9]
                ### token_abs_idx = [16, 17]
                ### abs_idx = 17


                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    entity_words = ent.split(" ")
                    #entity_words = list(self.tokenizer.encode(ent).tokens)[1:-1]

                    #ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(entity_words)+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    #ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    ent_abs_idx = [abs_idx + i for i in range(1, len(entity_words)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                    ## ent= "CEOApple"
                    ## len(ent) = 8
                    ### ent_pos_idx = [8, 9, 10, 11, 12, 13, 14, 15]
                    ### entities_pos_idx = [ [8, 9, 10, 11, 12, 13, 14, 15] ]
                    ### ent_abs_idx = [8, 9, 10, 11, 12, 13, 14, 15]
                    ### abs_idx = 15
                    ### entities_abs_idx = [ [8, 9, 10, 11, 12, 13, 14, 15] ]

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

                ## first step with token - "[CLS]"
                ### pos_idx_tree = [ ([0], []) ]
                ### pos_idx = 0
                ### abs_idx_tree = [ ([0], []) ]
                ### abs_idx_src = [0]
                ### abs_idx = 0 ## does not change

                ## second step with token - "Tim"
                ### pos_idx_tree = [ ([0], []), ([1, 2, 3], [])  ]
                ### pos_idx = 3
                ### abs_idx_tree = [ ([0], []), ([1, 2, 3], [])  ]
                ### abs_idx_src = [0, 1, 2, 3]
                ### abs_idx = 3 ## does not change

                ## third step with token - "Cook"
                ### pos_idx_tree = [ ([0], []), ([1, 2, 3], []), ([4, 5, 6, 7], [ [8, 9, 10, 11, 12, 13, 14, 15] ])  ]
                ### pos_idx = 7
                ### abs_idx_tree = [ ([0], []), ([1, 2, 3], []),  ([4, 5, 6, 7], [ [8, 9, 10, 11, 12, 13, 14, 15] ])   ]
                ### abs_idx_src = [0, 1, 2, 3, 4, 5, 6, 7]
                ### abs_idx = 15 ## changes inside the loop for entities management

                ## fourth step with token - "is"
                ### pos_idx_tree = [ ([0], []), ([1, 2, 3], []), ([4, 5, 6, 7], [ [8, 9, 10, 11, 12, 13, 14, 15] ]), ([8, 9],[])  ]
                ### pos_idx = 9
                ### abs_idx_tree = [ ([0], []), ([1, 2, 3], []),  ([4, 5, 6, 7], [ [8, 9, 10, 11, 12, 13, 14, 15] ]), ([16, 17],[])  ]
                ### abs_idx_src = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17]
                ### abs_idx = 17 ## does not change

            # Get know_sent and pos

            ### Suppose we have only four steps in the previous for loop:
            ### sent_tree = [("[CLS]",[]), ("Tim",[]), ("Cook",["CEO Apple"]), ("is",[])]

            #print(sent_tree)

            know_sent = []
            pos = []  
            seg = [] ### array with 1 for characters that belong to an entyty and 0 for charachters of the original sentence
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0] ## a special token is considered as a single character and is associated with a 0 because it comes from the original phrase and not from an entity
                else:
                    #add_word = list(word) #### we are dealing with english words and not chinese characters
                    add_word = word.split(" ") ## this should always be a single word but we use the split to create an array
                    # add_word = list(self.tokenizer.encode(word).tokens)[1:-1] # test vinc
                    know_sent += add_word 
                    seg += [0] * len(add_word) ## each character in the word is associated with a 0 because it comes from the original phrase and not from an entity
                pos += pos_idx_tree[i][0]

                ### i = 0
                #### know_sent = ["[CSL]"]
                #### seg = [0]
                #### pos = [0]

                ### i = 1
                #### know_sent = ['[CSL]', 't', 'i', 'm' ]
                #### seg = [0, 0, 0, 0]
                #### pos = [0,1,2,3]

                ### i = 2
                #### know_sent = ['[CSL]', 't', 'i', 'm', 'C', 'o', 'o', 'k' ]
                #### seg = [0, 0, 0, 0, 0, 0, 0, 0]
                #### pos = [0,1,2,3,4,5,6,7]
    
                
                #for j in range(len(words_in_entity)):
                for j in range(len(sent_tree[i][1])):
                    add_word = sent_tree[i][1][j].split(" ") ## Note: triples are already tokenized when tey are added to the lookup table
                    #add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

                ### i = 0
                #### know_sent = ["[CSL]"] ### does not change
                #### seg = [0]  ### does not change
                #### pos = [0]   ### does not change      
                 
                ### i = 1
                #### know_sent = ['[CSL]', 't', 'i', 'm' ] ### does not change
                #### seg = [0, 0, 0, 0]  ### does not change
                #### pos = [0,1,2,3]   ### does not change         
                 
                ### i = 2
                #### know_sent = ['[CSL]', 't', 'i', 'm', 'C', 'o', 'o', 'k', 'C', 'E', 'O', 'A', 'p', 'p', 'l', 'e' ] ### does not change
                #### seg = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  ### CEOApple is an entity and is marked with 1 in the seg array
                #### pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]                         


            token_num = len(know_sent)
            if debug:
                print("know_sent", know_sent)
                print("num tokens:", token_num)
            ### token_num = 16

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            ###we have a visibility matrix 16 x 16 because we are considering characters

            ### abs_idx_src = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17]
            ### abs_idx_tree = [ ([0], []), ([1, 2, 3], []),  ([4, 5, 6, 7], [ [8, 9, 10, 11, 12, 13, 14, 15] ]), ([16, 17],[])  ]


            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids: ### the token to wich triples are attached can see all other tokens in the original phrase plus the new tokens added with the triples attached
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent] ## a concatenation of character position of entities related to the item in the abdsolute indext tree of ids
                    visible_matrix[id, visible_abs_idx] = 1

                for ent in item[1]:
                    for id in ent: ## each token in the triple (es. "is a City") attached to one or more token (es "New York") can see all source tokens
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
                
                ## first step
                ## item = ([0], [])
                ### visible_matrix[0, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17] ] = 1 ## the first token [CLS] character can only see all characters from the original sentence                    
                ### no entities

                ## 
                ## item = ([1, 2, 3], [])
                ### visible_abs_idx = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17]
                ### visible_matrix[[1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 16, 17] ] = 1 ## the characters of the second token ("Tim") can only see all characters from the original sentence                    
                ### no entities   
                 
                ## 
                ## item = ([4, 5, 6, 7], [ [8, 9, 10, 11, 12, 13, 14, 15] ]
                ### visible_abs_idx = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 8, 9, 10, 11, 12, 13, 14, 15]
                ### visible_matrix[[4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 8, 9, 10, 11, 12, 13, 14, 15] ] = 1 ## the characters of the third token ("Cook") see all characters from the original sentence plus all the character from the etity asscoiated with the third token ("CEOApple")                    
                ### no entities
                         
            ###### Padding 
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
            num_triples_added_batch.append(num_triples_added)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch, sent_tree, num_triples_added_batch

