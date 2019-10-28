from __future__ import division
from Feature_Extractor import high_level_features
import tensorflow as tf
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from nltk.corpus import wordnet
import fasttext, nltk, os, pickle, sys, traceback
from sentenceTovector import sentTovec
import tensorflow_hub as hub
import re, cPickle, json, warnings
from sklearn.metrics import classification_report as cr
from Entity_Extractor import generate_entities
from pandas import DataFrame
import csv
from TopicClassifier_orig_hub import IntentClassifier

lmtzr = WordNetLemmatizer().lemmatize

if not sys.warnoptions:
    warnings.simplefilter("ignore")

####################################################
def get_wordnet_pos( treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

####################################################
def normalize_text(text):
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]

    return [x.lower() for x in lemm_words]



# sys.path.append(os.getcwd() + '/NLP/')

root_path = '/Users/aliahmadvand/Desktop/TopicClassifier/CNN_Concept/runs/1547155122/'

def set_state(state):
    new_state = state
    if state == '__label__no' or state == '__label__yes' or state == '__label__stop'  or state == 'unknown' or  state == '__special__no' or state == '__label__special':
        new_state = '__label__chitchat'
    elif state == '__label__cars':
        new_state = '__label__tech'
    elif state == '__label__worldcup':
        new_state = '__label__sports'
    else:
        pass

    return new_state


class IntentClassifier_main:
    def __init__(self):
        # with open(root_path + "dialogue_obj.pickle", 'r') as f:
        #     self.obj = pickle.load(f)
        # Eval Parameters
        tf.flags.DEFINE_string("checkpoint_dir",
                               root_path + "checkpoints",
                               "Checkpoint directory from training run")

        # Misc Parameters
        tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
        tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

        # self.stop = open(root_path + "stop_questions_final.txt", 'r').read().split("\n")
        self.FLAGS = tf.flags.FLAGS
        self.checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.FLAGS.allow_soft_placement,
                log_device_placement=self.FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)

            with self.sess.as_default():
                # Initlization !
                self.sess.run(tf.global_variables_initializer())

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                self.input_x_ent = self.graph.get_operation_by_name("input_entityvector").outputs[0]
                self.input_x_2 = self.graph.get_operation_by_name("input_x_hand").outputs[0]
                self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predict = self.graph.get_operation_by_name("output/scores").outputs[0]
                # self.lookup = self.graph.get_operation_by_name("embeddingc/WW").outputs[0]


        # self.graph_hub = tf.Graph()
        # with self.graph_hub.as_default():
        #     self.sess2 = tf.Session()
        #     self.embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
        #     with self.sess2.as_default():
        #         self.sess2.run(tf.global_variables_initializer())
        #         self.sess2.run(tf.tables_initializer())

        self.vocab_path = os.path.join(root_path, "..", "vocab")
        vocab_file = open('/Users/aliahmadvand/Desktop/TopicClassifier/CNN_Concept/runs/1547155122/vocab.json')
        self.vocab = json.load(vocab_file)
        filelabels = open('./auxiliary_files/class_order.json')
        self.labels_order = json.load(filelabels)
        entity_vector = dict()
        self.Types = {}
        with open('./entity_util/Type.json') as json_data:
            Types1 = json.load(json_data)
            for key in Types1.keys():
                self.Types[key.title()] = 0
        json_data.close()

        self.ent_dict = {}
        count = 1
        for key in self.Types.keys():
            self.ent_dict[key] = count
            count += 1

        self.rev_ent_dict = {}
        for key in self.ent_dict.keys():
            self.rev_ent_dict[self.ent_dict[key]] = key.lower()

        self.ent = generate_entities.Entity()
        print self.labels_order
        self.lmtzr = WordNetLemmatizer().lemmatize
        self.normalize_text("hi")


    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def normalize_text(self, text):
        word_pos = nltk.pos_tag(nltk.word_tokenize(text))
        lemm_words = [self.lmtzr(sw[0], self.get_wordnet_pos(sw[1])) for sw in word_pos]

        return [x.lower() for x in lemm_words]
    ################################################################################################
    def clean_str(self, string):
        cleaned_text = re.sub(r'<[^<]+?>', '', string)
        cleaned_text = re.sub(r'[a-z]*[:.]+\S+', '', cleaned_text)
        cleaned_text = " ".join(re.sub("[^A-Za-z0-9, ']+", ' ', cleaned_text).split()[0:19])
        cleaned_text = cleaned_text.replace(',', '')
        return cleaned_text.lower()

    def get_S2v_vector(self, dataset):
        with self.graph_hub.as_default():
            embeddings = self.embed(dataset)
            self.w2v_vectors = self.sess2.run(embeddings)

        # return self.w2v_vectors



    def entity_features(self, utt, length, entities):
        ent_features = np.zeros(length)
        # entities = self.ent.entity_extract(utt)
        ENTITY = []
        new_utt = utt

        for text in entities:
            position = []
            ent_typ = []
            ids = []
            for link in text['links']:
                for key in link:
                    types = link[key]['types']
                    for type in types:
                        if type['type'] in self.Types:
                            ENTITY.append(key)
                            position.append(text['position'][0])
                            ent_typ.append(type['type'])
                            ids.append(type['id'])
                            new_entity = '_'.join(key.lower().split())
                            new_utt = new_utt.replace(key.lower(), new_entity)

            # print utt, "    ", ent_typ
            for entity in ENTITY:
                new_ent_type = []
                for i, id in enumerate(ids):
                    if 'iris' in id:
                        new_ent_type.append(ent_typ[i])

                new_ent_type = ent_typ

                try:
                    if 'Animal' in new_ent_type:
                        ent_pos = []
                        words = entity.split()
                        for item, word in enumerate(words):
                            ent_features[position[ent_typ.index('Animal')] + item] = self.ent_dict['Animal']
                        break

                    if 'Country' in new_ent_type:
                        ent_pos = []
                        words = entity.split()
                        for item, word in enumerate(words):
                            ent_features[position[ent_typ.index('Country')] + item] = self.ent_dict['Country']
                        break


                    elif 'City' in new_ent_type and 'Place' in new_ent_type:
                        if 'Singers' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Singers')] + item] = self.ent_dict['Singers']
                            break
                        elif 'Actors' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Actors')] + item] = self.ent_dict['Actors']
                            break
                        elif 'Politicians' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Politicians')] + item] = self.ent_dict['Politicians']
                            break
                        else:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('City')] + item] = self.ent_dict['City']
                            break


                    elif 'Sports' in new_ent_type:
                        ent_pos = []
                        words = entity.split()
                        for item, word in enumerate(words):
                            ent_features[position[ent_typ.index('Sports')] + item] = self.ent_dict['Sports']
                        break


                    elif 'Singers' in new_ent_type:
                        ent_pos = []
                        words = entity.split()
                        for item, word in enumerate(words):
                            ent_features[position[ent_typ.index('Singers')] + item] = self.ent_dict['Singers']
                        break

                    elif 'Politicians' in new_ent_type and 'Singers' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Singers')] + item] = self.ent_dict['Singers']
                            break


                    elif 'Celebrities' in new_ent_type and 'Singers' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Singers')] + item] = self.ent_dict['Singers']
                            break

                    elif 'Politicians' in new_ent_type and 'Actors' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Actors')] + item] = self.ent_dict['Actors']
                            break

                    elif 'Celebrities' in new_ent_type and 'Actors' in new_ent_type:
                            words = entity.split()
                            for item, word in enumerate(words):
                                ent_features[position[ent_typ.index('Actors')] + item] = self.ent_dict['Actors']
                            break

                    else:
                        for i, pos in enumerate(position):
                            for t, typ in enumerate(ent_typ):
                                words = entity.split()
                                for item, word in enumerate(words):
                                    try:
                                       ent_features[pos + item] = self.ent_dict[ent_typ[t]]
                                    except:
                                        pass
                except:
                    pass

        new_ent_features = []
        for val in ent_features:
            if val > 0:
                new_ent_features.append(val)
            else:
                new_ent_features.append(0)


        # new_txt = utt
        # ent_vector = np.zeros(length)
        #
        # if len(best_entity) > 0:
        #     for entity in best_entity:
        #         ent = entity[0]
        #         new_entity = '_'.join(ent.split())
        #         new_txt = new_txt.replace(ent, new_entity)
        #
        #
        #
        #         #
        #         #
        #         # for ent in entities.keys():
        #         for i, word in enumerate(new_txt.split()):
        #             new_entity = '_'.join(ent.split())
        #             if word == new_entity:
        #                 type = entity[1]
        #                 # print entities[ent]
        #                 words = new_entity.split('_')
        #                 for j, wd in enumerate(words):
        #                     if type in self.ent_dict.keys():
        #                         ent_vector[i + j] = self.ent_dict[type]
        #                 break

        ent_features = np.array(ent_features).reshape(1, length)
        return ent_features, new_utt

    def hand_features(self, best_entity):

        handcarft_features = []
        # query, tokens = self.obj.query_preprocessing(utt, self.obj.dictionary, self.obj.stoplist)
        # handcarft_features.extend(self.w2v_vectors[i])
        # tfidf_vector = self.obj.Tfidf_Extractor(self.obj.bow_corpus, self.obj.query)
        # handcarft_features.extend(tfidf_vector)
        #
        # matrix_similarity = self.obj.Matrix_Similarity(self.obj.bow_corpus, self.obj.dictionary, query, self.obj.Labels)
        # handcarft_features.extend(matrix_similarity)
        # # lsi_similarity = self.obj.LSI_Similarity(self.obj.bow_corpus, self.obj.dictionary, query, self.obj.Labels)
        # # handcarft_features.extend(lsi_similarity)
        # try:
        #     lda_similarity = self.obj.Lda_Similarity(self.obj.bow_corpus, self.obj.dictionary, query, self.obj.Labels)
        # except:
        #     lda_similarity = np.zeros(15)
        # #
        # handcarft_features.extend(lda_similarity)

        entities = self.ent.entity_extract(utt)
        ent_vector = np.zeros(len(self.Types.keys()))
        for text in entities:
            position = []
            ent_typ = []
            ids = []
            for link in text['links']:
                for key in link:
                    types = link[key]['types']
                    for type in types:
                        if type['type'] in self.Types:
                            try:
                                ent_vector[self.ent_dict[type['type']]] = 1
                            except:
                                pass

        # G = list(np.zeros(4))
        # if 'music' in utt or 'sing' in utt or 'sang' in utt:
        #     G[0] = 1
        # elif 'movie' in utt or 'film' in utt:
        #     G[1] = 1
        # elif 'news' in utt:
        #     G[2] = 1
        # elif 'sport' in utt:
        #     G[3] = 1
        # else:
        #     pass
        #
        # ent_vector = np.zeros(len(self.Types.keys()))
        # if len(best_entity) > 0:
        #     try:
        #         if best_entity[1] in self.Types:
        #             try:
        #                 ent_vector[self.ent_dict[type['type']]] = 1
        #             except:
        #                 pass
        #     except:
        #
        #         f = 0


        res = list(ent_vector)
        handcarft_features = np.array(res).reshape(1, len(self.Types))
        return handcarft_features, entities

    def prediction_step(self, x_test, handcarft_features, ent_vect):
        x_p = np.array(self.sess.run(self.predict, {self.input_x: x_test, self.input_x_2: handcarft_features, self.input_x_ent: ent_vect, self.dropout_keep_prob: 1.0}))
        # x_p3 = np.array(self.sess.run(self.lookup, {self.input_x: x_test, self.input_x_2: handcarft_features, self.input_x_ent: ent_vect, self.dropout_keep_prob: 1.0}))
        normalized_prediction = (np.exp(x_p)) / np.sum(np.exp(x_p))


        final_prediction = []
        for i, label in enumerate(self.labels_order):
            label_score = ( self.labels_order[label], normalized_prediction[0][int(label)])
            final_prediction.append(label_score)
        final_prediction.sort(key=lambda x: x[1], reverse=True)
        return final_prediction


    # def classify(self, utterance, contextInfo):
    def getObject(self, utterance):
        h_features, entities = self.hand_features(utterance)
        # x_test = np.array(list(self.vocab_processor.transform([lemm_text])))
        len =  self.vocab['max_doc_len']
        ent_vec, new_utt = self.entity_features(utterance, len, entities)
        x_test = np.zeros(len)
        for i, word in enumerate(new_utt.split()):
            try:
                x_test[i] = self.vocab['data'][word]
            except:
                pass

        x_test = list(np.reshape(x_test,[1,len]))

        # h_features = np.array(handcraft + h_features1).reshape(1, 4)
        intent_prob = self.prediction_step(x_test, h_features, ent_vec)
        intent = intent_prob[0][0]
        # print intent_prob
        return intent_prob


file_pred = open('prediction.txt', 'w')
# # dataset_path = '/Users/aliahmadvand/Desktop/TopicClassifier/cnn-tf-handcraft-training_Whole/test_utterances_newclasses.txt'
# # contents = open( dataset_path, 'r').read().split("\n")
#
# dataset_path = 'test_real_data_utterances.txt'
# contents_ = open(dataset_path, 'r').read().split("\n\n")
# contents = []
# for dialogue in contents_:
#     data = dialogue.split('\n')
#     contents.extend(data)


correct = 0
y_pred = []
y_test = []
ground_truth = []
predicted_label = []
predicted_label_TM = []

# #initializes the classifier
Classifier = IntentClassifier_main()
#
# utts = []
# for i, text in enumerate(contents):
#     if len(text) > 0:
#         utts.append(text.split('\t')[1].strip())
#         labels = text.split('\t')[0].strip()

# Classifier.get_S2v_vector(utts)
# contents = ['__label__tech\ti want to talk about robots', '__label__tech\ti want to talk about electric machines',
#              '__label__tech\ti want to talk about programing with c', '__label__sports\t lets chat about Skiing']
# For each test sample
label_list = ['__label__music', '__label__attraction', '__label__news', '__label__games','__label__movie', '__label__pets_animals',
                  '__label__literature', '__label__weather', '__label__sports', '__label__food', '__label__joke', '__label__chitchat',
                  '__label__tech', '__label__other', '__label__celebrities', '__label__fashion'
                 ]

yes_no = []
utts = []
indexes = []
with open('/Users/aliahmadvand/Desktop/MixtureofExperts/AmazonAnnotator/ali_labels.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for i, row in enumerate(csv_reader):
        if set_state(row['decision']) in label_list:
            yes_no.append((row['predicted']))
        else:
            if set_state(row['decision']) != '__label__opening':
                # print row['decision']
                pass
#
contents = open('/Users/aliahmadvand/Desktop/TopicClassifier/CNN_Concept/alexa_entity_data/test_data_el.txt', 'r').read().split('\n')


with open('/Users/aliahmadvand/Desktop/TopicClassifier/generate_dataset/classifier/pickle_files/hand_hub_dataset_test.pickle','rb') as handle:
    data_handcraft = cPickle.load(handle)


dataset_withoutcontext = open('/Users/aliahmadvand/Desktop/TopicClassifier/CNN_Contextual/Generate_dataset/Final_datasets/real_dialogue_test_context.txt', 'r').read().split('\n')

utts = []
real_labels = []

for line in dataset_withoutcontext:
    if len(line) > 0 :
        real_labels.append(line.split('\t')[0])
        utts.append(line.split('\t')[1])



data_wrong = open('wrong.txt', 'w')
classification = []
j = -1
for label, utt in zip(real_labels,utts):
    # if len(sample1) > 0:

        # sample = json.loads(sample1)
        # utt = sample['utterances'][-1]
        # label = set_state(sample['final_state'][-1])
        # best_entity = sample['best_entities'][-1]
        # print utt

        if label in label_list:


                    # print utt, '    '    ,label
                    new_utt = ' '.join (utt.split()[0:19])
                    check =0
                    # OwnIntentOrig = OwnClassifier.getclassOrig(utt)


                    intent = Classifier.getObject(Classifier.clean_str(new_utt))
                    if 'pokemon' in new_utt or 'minecraft' in new_utt or 'fortnite'in new_utt or 'nintendo' in new_utt:
                        intent[0] = ('__label__games', 1)
                    if 'travel' in new_utt:
                        intent[0] = ('__label__attraction', 1)
                    if 'harry potter' in new_utt:
                        intent[0] = ('__label__literature', 1)

                    # if new_utt == 'yes' or new_utt == 'yeah' or new_utt == 'sure' or new_utt == 'okay sure' or new_utt == 'okay' or new_utt == 'no' or new_utt == 'stop':
                    #     intent[0] = ('__label__chitchat', 1)
                    #     check = 1

                    print utt, '   ', intent[0][0]
                    prediction = intent.pop(0)

                    y_pred.append(prediction[0])
                    y_test.append(label)

                    # if utt == 'yes' or utt == 'yeah' or utt == 'sure':
                    #     pre_class_name = 'chitchat'

                    if prediction[0] != label:
                        line = utt + "\t\t" + label + "\t\t" + "(" + prediction[0] + ", " + str(prediction[1]) + ")"
                        data_wrong.write(line)
                        data_wrong.write("\n")



                    line  = utt + "\t\t" + label + "\t\t" + "(" + prediction[0] + ", " + str(prediction[1]) + ")"
                    file_pred.write(line)
                    file_pred.write("\n")
                    g =  '-' * 100
                    file_pred.write(g + "\n")

                    ###### F1 score for each class
                    class_name = label.split('__')[-1].strip()
                    pred_class_name = prediction[0].split('__')[-1].strip()
                    if len(pred_class_name) == 0:
                        g = 0




                    # if true_label != 'chitchat' and map_pred != 'chitchat':

                    true_label = class_name


                    if class_name == pred_class_name:
                        map_pred = class_name
                    elif class_name == 'movie' or class_name == 'music':
                         if pred_class_name == 'celebrities':
                             map_pred = class_name
                    elif pred_class_name == 'movie' or pred_class_name == 'music':
                        if class_name == 'celebrities':
                            map_pred = class_name
                    else:
                        map_pred = pred_class_name


                    if class_name == pred_class_name:
                       classification.append(1)
                    else:
                        classification.append(0)



                    ground_truth.append(class_name)
                    predicted_label.append(map_pred)
                    predicted_label_TM.append(pred_class_name)

                    passclasses = ['__label__movie', '__label__music', '__label__attraction', '__label__pets_animals',
                                   '__label__games', '__label__news', '__label__sports']
                    if len(predicted_label) > 1:
                        if '__label__' + predicted_label[-2] in passclasses and check == 1:
                            predicted_label_TM[-1] = predicted_label[-2]


accuracy = 0
for i, item in enumerate(ground_truth):
    if predicted_label[i] == item:
        accuracy += 1


print "Accuracy is :   ", accuracy/len(predicted_label)

print cr(ground_truth, predicted_label, digits=3)

# print "Transition Matrix"
#
# print cr(ground_truth, predicted_label_TM, digits=3)

table =  cr(ground_truth, predicted_label, digits=3)

rows = table.split('\n')
c = []
p = []
r = []
f = []
s = []

for i, sample in enumerate(rows):

    if i >= 2:
        values = sample.split()
        if len(values) > 0:
            c.append(values[0])
            p.append(values[1])
            r.append(values[2])
            f.append(values[3])
            s.append(values[4])

df = DataFrame({'class': c, 'precision': p, 'recall': r, 'fscore': f,'support': s})
df.to_excel('result.xlsx', sheet_name='sheet1', index=False)

g = 0
from sklearn.metrics import f1_score, precision_score, recall_score
print precision_score(ground_truth, predicted_label, average='micro')
print recall_score(ground_truth, predicted_label, average='micro')
print f1_score(ground_truth, predicted_label, average='micro')

print
print "===============Macto==============="
print

print precision_score(ground_truth, predicted_label, average='macro')
print recall_score(ground_truth, predicted_label, average='macro')
print f1_score(ground_truth, predicted_label, average='macro')

file_data = open('classfication_cet.txt','w')
for label in classification:
    file_data.write(str(label))
    file_data.write('\n')

