import numpy as np


class ModelUtils:
    @staticmethod
    def build_text_image(word2vec, words, embedding_size=300):
        word_vectors = []

        for w in words:
            w_vec = word2vec.get_word_vector(w)
            if w_vec is None:
                word_vectors.append(np.array([0.] * embedding_size))
            else:
                word_vectors.append(w_vec)

        return [word_vectors, ]

    @staticmethod
    def mrr(q_list, y_true, y_pred):
        assert sum(len(i) for i in q_list) == len(y_pred)

        index_begin = 0
        mrr = 0.0

        for q in q_list:
            index_end = index_begin + len(q)
            predictions_slice = y_pred[index_begin:index_end].flatten().tolist()

            # Pair label-prediction and sort by prediction descending order
            xx = zip(q, predictions_slice)
            xx = sorted(xx, key=lambda tup: tup[1], reverse=True)

            for idx, x in enumerate(xx):
                if x[0] == 1:
                    mrr += float(1) / (idx + 1)
                    break

            index_begin = index_end

        mrr = float(mrr) / len(q_list) * 100
        return mrr

    @staticmethod
    def map(q_list, y_true, y_pred):
        avg_prec = 0
        index_begin = 0

        for q in q_list:
            index_end = index_begin + len(q)
            predictions_slice = y_pred[index_begin:index_end].flatten().tolist()

            correct_answers = len([1 for x in q if x == 1])

            xx = zip(q, predictions_slice)
            xx = sorted(xx, key=lambda tup: tup[1], reverse=True)

            correct = 0
            wrong = 0
            av_prec_i = 0

            for idx, x in enumerate(xx):
                if x[0] == 1:
                    correct += 1
                else:
                    wrong += 1

                if x[0] == 1:
                    av_prec_i += float(correct) / (correct + wrong)

                if correct == correct_answers:
                    break

            if correct_answers > 0:
                avg_prec += av_prec_i / correct_answers

            index_begin = index_end

        omap = float(avg_prec) / len(q_list) * 100
        return omap

    @staticmethod
    def precision_recall_f1(q_list, y_true, y_pred):
        results = []

        for thr in range(1, 15):
            thr /= 100.0
            p, r, f1 = ModelUtils._precision_recall_f1_threshold(q_list, y_true, y_pred, thr)
            results.append(('thre: %.2f, prec: %.2f, rec: %.2f, f1: %.2f' % (thr, p, r, f1), thr, p, r, f1))

        return results

    @staticmethod
    def _precision_recall_f1_threshold(q_list, y_true, y_pred, thre):
        index_begin = 0
        all_questions_with_answers = 0.
        predicted_questions = 0.
        correctly_predicted_questions = 0.

        for q in q_list:
            # Get the slice from predictions
            index_end = index_begin + len(q)
            predictions_slice = y_pred[index_begin:index_end].flatten()

            # Find the maximum value prediction
            max_val_index = np.argmax(predictions_slice)

            if predictions_slice[max_val_index] > thre:
                predicted_answer = max_val_index
            else:
                predicted_answer = -1

            # Check if this question has an answer.
            gold_answer_ids = [x for x, y in enumerate(q) if y == 1]
            nb_gold_correct_answers = len(gold_answer_ids)

            # If there is an answer, increment number of all questions
            if nb_gold_correct_answers > 0:
                all_questions_with_answers += 1

            # If the question predicted with the answer, increment predicted_questions
            if predicted_answer > -1:
                predicted_questions += 1

            # If the question predicted correctly, increment correctly_predicted_questions
            if predicted_answer > -1 and predicted_answer in gold_answer_ids:
                correctly_predicted_questions += 1

            index_begin = index_end

        if correctly_predicted_questions == 0 or predicted_questions == 0:
            return 0., 0., 0.

        precision = float(correctly_predicted_questions) / predicted_questions * 100
        recall = float(correctly_predicted_questions) / all_questions_with_answers * 100
        f1 = (2 * precision * recall) / (precision + recall) if (precision > 0.0 and recall > 0.0) else 0.0

        return precision, recall, f1
