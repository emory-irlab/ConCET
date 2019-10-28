import numpy as np
import re, json, nltk, os
import itertools
from collections import Counter
import cPickle
from embedding import Word2Vec
from time import time
from copy import deepcopy
from collections import defaultdict
import time
from sklearn.utils import shuffle
from collections import Counter
import random


root = os.path.dirname(os.getcwd())
root = root + '/ConCET'
# root = root + '/EntityClassifier'


pos_dict = {'CC': 1, 'CD': 2, 'DT': 3, 'EX': 4, 'FW': 5, 'IN': 6, 'JJ': 7, 'JJR': 8, 'JJS': 9,
            'LS': 10, 'MD': 11, 'NN': 12, 'NNS': 13, 'NNP': 14, 'NNPS': 15, 'PDT': 16, 'POS': 17, 'PRP': 18,
            'PRP$': 19, 'RB': 20, 'RBR': 21, 'RBS': 22, 'RP': 23, 'SYM': 24, 'TO': 25, 'UH': 26, 'VB': 27, 'VBD': 28,
            'VBG': 29, 'VBN': 30, 'VBP': 31, 'VBZ': 32, 'WDT': 33, 'WP': 34, 'WP$': 35, 'WRB': 36, ',': 37, "''": 38, 'OTHER': 39}


with open(root + '/datasets/Spotlight/final_version/self_dialogue/spotlight_entity_dict.json') as json_data:
    spotlight_types = json.load(json_data)
json_data.close()

################################################################################################
#vector to respresent the onehot vector for entities
entity_vector = dict()
Types = {}

# for Spotlight this one has to be From ***DbPedia**** Folder
with open(root + '/datasets/Dbpedia_database/dbpedia_types.json') as json_data:
    Types1 = json.load(json_data)
    for key in Types1.keys():
        Types[key.strip()] = 0
json_data.close()

ent_dict = {}
count  = 1
for key in Types.keys():
    ent_dict[key] = count
    count += 1

################################################################################################
def clean_str(string):
    cleaned_text = re.sub(r'<[^<]+?>', '', string)
    cleaned_text = re.sub(r'[a-z]*[:.]+\S+', '', cleaned_text)
    cleaned_text = " ".join(re.sub("[^A-Za-z0-9, ']+", ' ', cleaned_text).split())
    cleaned_text = cleaned_text.replace(',', '')
    return cleaned_text.lower()

################################################################################################
def char2vec(text, sequence_max_length, char_dict):
    data = np.zeros(sequence_max_length)
    for i in range(0, len(text)):
        if i > sequence_max_length:
            return data
        elif text[i] in char_dict:
            try:
                data[i] = char_dict[text[i]]
            except:
                pass
        else:
            # unknown character set to be 68
            try:
                data[i] = 39
            except:
                pass
    return data
################################################################################################
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789' "
char_dict = {}
for i, c in enumerate(alphabet):
    char_dict[c] = i + 1

def load_char_text(utterance, sequence_max_length):
    return char2vec(utterance, sequence_max_length, char_dict)

################################################################################################
def load_data_and_labels(train_file, test_file):

    training_set = open(train_file).read().split('\n')
    test_set =  open(test_file).read().split('\n')

    # creating Vocabulary
    vocab, w2v, data_entity, All_onehot_vector, new_data, train_test_dev, new_label, handcrafted_text, data_pos, data_char = create_vocabulary(training_set, test_set, train_file, test_file)
    # one_hot_word_vector = generate_onehot_word_vector()
    # one_hot_sentence_vector = generate_onehot_vector(All_sentence_vector)

    #preparing data for training
    num_classes = 17
    data_label = []
    data = []
    class_label = dict()
    classl = -1

    # with open(train_file) as f:
    for iter, line in enumerate(new_label):
        # split_line = line.strip().split('\t')
        label = line
        if len(label) > 0 :
            text = line
            data.append(text)
            one_hot = np.zeros(num_classes)
            if label not in class_label.keys():
                classl = classl + 1
                print label
                # print classl
                class_label[label] = classl
            one_hot[class_label[label]] = 1
            data_label.append(one_hot)

    class_order = dict()
    for label in class_label:
        class_order[class_label[label]] = label

    file_classorder = open('./auxiliary_files/class_order.json', 'w')
    json.dump(class_order, file_classorder)

    # adding BAG oF ENTITIES (BOE)
    bag_of_entity = []
    for i, sample in enumerate(All_onehot_vector):
        # Bag_of_entity.append(np.ones(1))
        bag_of_entity.append(sample)
        # print handcraft_train[i]

    return np.array(new_data), np.array(data_char),np.array(data_label), np.array(handcrafted_text), np.array(bag_of_entity), vocab, w2v, data_pos, data_entity, pos_dict, ent_dict, train_test_dev, class_label

################################################################################################
def create_vocabulary(training_set, test_set,  train_file, test_file):

    # generate a huge vocab
    count = 0
    vocabulary = set()  # to save the vocabulary
    start = time.time()
    new_data = []
    new_label = []
    All_sentence_vector = []
    All_onehot_vector = []
    data_pos = []

    vocabulary = set()
    # chitchat = open('/Users/aliahmadvand/Desktop/TopicClassifier/CNN_Concept/handcrafted_features/train_selfdialogue.txt').read().split('\n')
    # for utt in chitchat:
    #     vocabulary.update(utt.split())

    with open(root + '/handcrafted_features/self_dialogue_train_td', 'rb') as handle:
        self_dialogue_train_td = cPickle.load(handle)
    with open(root + '/handcrafted_features/self_dialogue_test_td', 'rb') as handle:
        self_dialogue_test_td = cPickle.load(handle)

    entity_dict = {}
    ii = 0
    train_test_dev = 0
    samples = 0
    handcrafted_text = []
    data_char = []

    # k = int (len(training_set) *  0.05)
    # indicies = random.sample(xrange(len(training_set)), k)
    # new_train = [training_set[i] for i in indicies]


    # synthetic = open(root + "/datasets/Spotlight/final_version/synthetic_dialogue/train_synthetic_dialogue_adan_spotlight.txt").read().split('\n')
    # sync_data = []
    # for line in synthetic:
    #     if len(line) > 0:
    #         line1 = json.loads(line)
    #         label = line1['label']
    #         if label == "__label__movie" or label == "__label__music" or label == "__label__sports" or label == "__label__fashion" :
    #            sync_data.append(line)
    #         elif label == "__label__chitchat":
    #             # pass
    #             vocabulary.update(line1['utterance'].split())


    # new_train = []
    # new_train.extend(sync_data)
    training_set = shuffle(training_set)

    dialogue_dataset = [training_set, test_set]
    for iter, dataset in enumerate(dialogue_dataset):
        for i, line in enumerate(dataset):
            if (iter == 0 and i % 1 == 0) or iter == 1:
                if iter == 0 and i > len(training_set):
                    print "  im inside....."
                    break

                if len(line) > 0:


                    sample = json.loads(line)
                    text = sample['utterance']
                    if iter == 1 and samples == 0:
                        train_test_dev = len(new_data) - 1
                        samples += 1
                        print "train_test_dev: ", train_test_dev

                    new_txt = clean_str(text.lower())
                    orig_txt = clean_str(text.lower())
                    entities = sample['entities']
                    if iter == 0:
                       vocabulary.update(new_txt.split())
                    lbl = sample['label'].split('__')[-1].strip()
                    adan_lbl =  sample['adan_label']


                    # if ((iter == 0 and lbl != 'chitchat') and (iter == 0 and lbl != 'other')) or ((iter == 1 and adan_lbl != 'Phatic') and (iter == 1 and lbl != 'chitchat')):
                    if adan_lbl != 'Phatic':

                        entity_to_type = {}
                        entity_lists = {}
                        if len(entities) > 0:
                            ent_verctor = []
                            for entity_ in entities:
                                entity = entity_['entity'].lower()
                                new_entity = '_'.join(entity.split())
                                new_txt = new_txt.replace(entity, new_entity)
                                entity_to_type[entity] = entity_['best_type']
                                entity_lists.update(entity_['entity_list'])

                            # print iter, new_txt

                            Sectence_ent_vector = get_entity_vector(orig_txt, new_txt, entity_to_type)
                            pos_vector = getting_posentity_features(Sectence_ent_vector, orig_txt, new_txt, pos_dict)



                            one_hot_vector = get_onehot_entity_vector(orig_txt, new_txt, entity_lists)
                            new_data.append(orig_txt)
                            new_label.append(lbl)
                            All_sentence_vector.append(Sectence_ent_vector)
                            All_onehot_vector.append(one_hot_vector)
                            data_pos.append(getting_extra_features(orig_txt, pos_dict))
                            data_char.append(load_char_text(text, 50))
                            entity_dict[orig_txt] = Sectence_ent_vector

                            if iter == 0:
                                handcrafted_text.append(np.ones(1))
                            else:
                                handcrafted_text.append(np.ones(1))

                        else:
                            new_data.append(orig_txt)
                            new_label.append(lbl)
                            All_sentence_vector.append([0])
                            All_onehot_vector.append(np.zeros(len(spotlight_types.keys())))
                            data_pos.append(getting_extra_features(orig_txt, pos_dict))
                            data_char.append(load_char_text(text, 50))
                            entity_dict[orig_txt] = [0]

                            if iter == 0:
                                handcrafted_text.append(np.ones(1))
                            else:
                                handcrafted_text.append(np.ones(1))




    print "=======================> {}".format(len(vocabulary))

    print time.time() - start
    with open('./auxiliary_files/vocabulary.pkl', 'wb') as handle:
        cPickle.dump(vocabulary, handle)


    server = '/data/ali/Semantic-Clustering/word2vec.bin'
    local = '/Users/aliahmadvand/Documents/AlexaPrize2018/Data/Word2Vec.bin'
    w2v = Word2Vec(local, vocabulary)
    # w2v.save_model('./auxiliary_files/Word2Vector.model')
    #
    # print len(w2v.word_vectors.keys())

    # with open('./auxiliary_files/Entity_Vectors.json', 'wb') as handle:
    #     json.dump(entity_dict, handle)


    vocab = {}
    for item in vocabulary:
        vocab[item] = count
        count += 1

    print "=======================> {}".format(len(vocab))

    return vocab, w2v, All_sentence_vector, All_onehot_vector, new_data, train_test_dev, new_label, handcrafted_text, data_pos, data_char


################################################################################################
# Assign onehot vector for every word and entity in the dictionary
def get_entity_vector(orig_txt, new_text, entities):
    if len(entities.keys()) > 1:
        f = 0

    orig_txt = orig_txt.replace("'s", " s")
    ent_vector = np.zeros(len(orig_txt.split()))
    word_orig = orig_txt.split()

    try:

        for ent in entities.keys():
            new_entity = ent.split()
            start = new_entity[0]
            position = word_orig.index(start)
            type_ = entities[ent]
            for j, wd in enumerate(new_entity):
                if type_ in ent_dict.keys():
                     ent_vector[position + j] = ent_dict[type_]
    except:
        d =0



    return list(ent_vector)

################################################################################################
def get_onehot_entity_vector(orig_txt, new_text, entities):

    ent_vector = np.zeros(len(spotlight_types.keys()))
    for entity in entities:
       try:
          ent_vector[spotlight_types[entity]] = entities[entity]
       except:
           pass

    return ent_vector



################################################################################################
def getting_posentity_features(Sectence_ent_vector, orig_txt, new_txt, pos_dict):

    orig_txt = orig_txt.replace("'s", " s")
    pos_tags_new = nltk.pos_tag(new_txt.split())
    pos_tags_orig = nltk.pos_tag(orig_txt.split())
    word_orig = orig_txt.split()
    vector_pos = list(np.ones(len(Sectence_ent_vector))* pos_dict['OTHER'])


    try:
        if '_'  in  new_txt:
            new_txt_words = new_txt.split()
            for i, word in enumerate(new_txt_words):
                if '_' in word:
                    pos_tag_ent = pos_tags_new[i][1]
                    ent_words = word.split('_')
                    start = ent_words[0]
                    position = word_orig.index(start)

                    for j, wd in enumerate(ent_words):
                        vector_pos[position + j] = pos_dict[pos_tag_ent]


        else:
            vector_pos = []
            for i, item in enumerate(Sectence_ent_vector):
                if item != 0:
                   vector_pos.append(pos_dict[pos_tags_orig[i][1]])
                else:
                    vector_pos.append(pos_dict['OTHER'])
    except:
        vector_pos = list(np.ones(len(word_orig))* pos_dict['OTHER'])


    # if new_txt.count('_') > 1:
    #     print new_txt, '   ', vector_pos

    return vector_pos


################################################################################################
def generate_onehot_word_vector():
    features = defaultdict(list)
    for ent in entity_vector:
        for type in entity_vector[ent]:
            # print type
            features[ent].append(entity_vector[ent][type])

    return features

def generate_onehot_vector(dataset_onehot_dict):
    features = []
    for sample in dataset_onehot_dict:
        vector = []
        for type in sample:
            vector.append(sample[type])

        features.append(vector)

    return features


################################################################################################
def reset_type(types):
    for key in types.keys():
        types[key] = 0

    return types


################################################################################################
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]






################################################################################################
def fit_transform(train_data, vocab_processor, vocab):

    count = 0
    for iter, line in enumerate(train_data):
        text = line.strip()
        for j, word in enumerate(text.split()):
            try:
                vocab_processor[iter][j] = vocab[word]
            except:
                pass


    return vocab_processor


def fit_transform_pos(extra_pos_features, vocab_processor_pos):
    for iter, line in enumerate(extra_pos_features):
        for j, sample in enumerate(line):
            try:
               vocab_processor_pos[iter][j] = sample
            except:
                pass

    return vocab_processor_pos


def getting_extra_features(orig_txt, pos_dict):
    pos_tags_orig = nltk.pos_tag(orig_txt.split())
    word_pos = map(lambda x: (x[1]), pos_tags_orig)
    vector_pos = []
    for pos in word_pos:
        try:
           vector_pos.append(pos_dict[pos])
        except:
            vector_pos.append(0)

    return vector_pos
