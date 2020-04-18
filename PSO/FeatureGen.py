import math
import numpy as np
from collections import defaultdict
from .constants import *

class FeatureGen:
    def __init__(self, chapter="", n_grams=2, summary_len=75):
        # chapter = prep-processed text doc
        self.text = chapter
        self.summary_length = summary_len
        self.n_grams = n_grams
        # let this be lists of lists, where each feature value is stored in a list with indices corresponding to sentence number
        self.feature_vector_list = []
        # For calculating word frequency in sentence
        self.sentence_frequency_dict = dict()
        self.sentence_similarity_dict = dict()
        self.shared_friends_dict = dict()

    def setdoc(self, new_chapter):
        self.text = new_chapter

    def set_n_grams(self, new_ngram):
        self.n_grams = new_ngram

    def set_summary_length(self, new_sum_len):
        self.summary_length = new_sum_len

    def create_ngrams(self, tokens, n):
        ngrams = defaultdict(int)
        for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    def get_Topic_Sentence_Feature(self):

        title = ''
        title_ngrams = set()

        feature_val_list = []

        for sentence in self.text:
            first_word = sentence[0]
            if first_word == '@':
                title_tokens = sentence[1:]
                title_2grams = list(self.create_ngrams(title_tokens, 2).keys())
                title_3grams = list(self.create_ngrams(title_tokens, 3).keys())
                title_ngrams = set(title_2grams + title_3grams)
                # print(title_ngrams)

            else:

                sent_2grams = list(self.create_ngrams(sentence, 2).keys())
                sent_3grams = list(self.create_ngrams(sentence, 3).keys())
                sent_ngrams = set(sent_2grams + sent_3grams)
                feature_val = len(sent_ngrams.intersection(title_ngrams)) / len(sent_ngrams.union(title_ngrams))
                feature_val_list.append(feature_val)
                # print( sent_ngrams)
        return feature_val_list

    def term_frequency(self, sentence):
        """
        Args:
            sentence (List[str]): Sentence whose term frequency is to be determined

        Returns:
            frequencies (dict): Returns term frequency for each term in sentence
        """
        frequencies = dict()

        for term in sentence:
            if term in frequencies:
                frequencies[term] += 1
            else:
                frequencies[term] = 1

        for key in frequencies.keys():
            frequencies[key] = frequencies[key] / len(sentence)

        return frequencies
                    
    def sentence_frequency(self):

        self.sentence_frequency_dict["length"] = len(self.text)

        for sentence in self.text:
            words = set(sentence)
            for word in words:
                if word in self.sentence_frequency_dict:
                    self.sentence_frequency_dict[word] += 1
                else:
                    self.sentence_frequency_dict[word] = 1

    def inverse_sentence_frequency(self, word):
        """
        Args:
            word (str): Given word
            
        Returns:
            score (float): ISF score
        """
        n = self.sentence_frequency_dict["length"]
        numerator = math.log(self.sentence_frequency_dict[word] + 1)
        denominator = math.log(n + 1)

        score = (1 - numerator / denominator) ** 2
        return score

    def calculate_similarity_sentences(self, sentence1, sentence2):
        """
        Args:
            sentence1 (List[str]): Sentence one
            sentence2 (List[str]): Sentence two
            
        Returns:
            score (float): Similarity score between sentence1 and sentence2
        """
        sentence1_string = ' '.join(sentence1)
        sentence2_string = ' '.join(sentence2)
        if sentence1_string in self.sentence_similarity_dict:
            if sentence2_string in self.sentence_similarity_dict[sentence1_string]:
                return self.sentence_similarity_dict[sentence1_string][sentence2_string]
        else:
            self.sentence_similarity_dict[sentence1_string] = dict()
        tf_sentence1_dict = self.term_frequency(sentence1)
        tf_sentence2_dict = self.term_frequency(sentence2)

        words1 = set(sentence1)
        words2 = set(sentence2)

        numerator = 0.0
        denominator1 = 0.0
        denominator2 = 0.0
        for word1 in words1:
            if word1 in words2:
                numerator += tf_sentence1_dict[word1] * tf_sentence2_dict[word1] * self.inverse_sentence_frequency(
                    word1)
                denominator1 += tf_sentence1_dict[word1] * self.inverse_sentence_frequency(word1)
                denominator2 += tf_sentence2_dict[word1] * self.inverse_sentence_frequency(word1)

        denominator = (denominator1 ** 0.5) * (denominator2 ** 0.5)
        if denominator == 0.0:
            self.sentence_similarity_dict[sentence1_string][sentence2_string] = 0.0
            return 0.0

        score = numerator / denominator
        self.sentence_similarity_dict[sentence1_string][sentence2_string] = score
        
        return score

    def calculate_similarity(self, sentence):
        """
        Args:
            sentence (List[str]): Sentence whose centrality is to be calculated
            
        Returns:
            score (float): Sum of similarity between sentence and all other sentences in document
        """
        score = 0.0

        for individual_sentence in self.text:
            if individual_sentence == sentence or individual_sentence[0] == '।@':
                continue
            score += self.calculate_similarity_sentences(sentence, individual_sentence)

        return score

    def shared_grams_sentences(self, sentence1, sentence2):
        """
        Args:
            sentence1 (List[str]): Sentence one
            sentence2 (List[str]): Sentence two
            
        Returns:
            score (float): Shared gram score between sentence1 and sentence2
        """
        words_sentence1 = set(self.create_ngrams(sentence1, 1).keys())
        words_sentence2 = set(self.create_ngrams(sentence2, 1).keys())

        numerator = len(words_sentence1.intersection(words_sentence2))
        denominator = len(words_sentence1.union(words_sentence2))

        if denominator == 0:
            return 0
        
        score = numerator / denominator

        return score

    def calculate_shared_gram_score(self, sentence):
        """
        Args:
            sentence (List[str]): Sentence whose shared_gram_score is to be calculated
            
        Returns:
            score (float): Sum of similarity between sentence and all other sentences in document
        """
        score = 0.0

        for individual_sentence in self.text:
            if individual_sentence == sentence or individual_sentence[0] == '@':
                continue
            score += self.shared_grams_sentences(sentence, individual_sentence)

        return score

    def friends_sentences(self, sentence1, sentence2):
        """
        Args:
            sentence1 (List[str]): Sentence one
            sentence2 (List[str]): Sentence two
            
        Returns:
            score (float): Friends score between sentence1 and sentence2
        """
        # Template code so program compiles
        sentence1_string = ' '.join(sentence1)
        sentence2_string = ' '.join(sentence2)
        
        sentence1_friends = set(self.shared_friends_dict[sentence1_string])
        sentence2_friends = set(self.shared_friends_dict[sentence2_string])

        numerator = len(sentence1_friends.intersection(sentence2_friends))
        denominator = len(sentence1_friends.union(sentence2_friends))

        if denominator == 0:
            return 0
        score = numerator / denominator

        return score 

    def calculate_friends_score(self, sentence):
        """
        Args:
            sentence (List[str]): Sentence whose friends_score is to be calculated
            
        Returns:
            score (float): Sum of similarity between sentence and all other sentences in document
        """
        score = 0.0

        for individual_sentence in self.text:
            if individual_sentence == sentence or individual_sentence[0] == '@':
                continue
            score += self.friends_sentences(sentence, individual_sentence)
        
        return score

    def sentence_centrality(self):
        """
        Returns:
            feature_val_list (List[float]): Sentence centrality feature values
        """
        feature_val_list = []
        n = len(self.text)
        for sentence in self.text:
            if sentence[0] == '@':
                continue
            similarity_score = self.calculate_similarity(sentence)
            shared_grams_score = self.calculate_shared_gram_score(sentence)
            shared_friends_score = self.calculate_friends_score(sentence)
            #print(similarity_score, shared_grams_score, shared_friends_score)
            score = (similarity_score + shared_grams_score + shared_friends_score) / (n - 1)
            feature_val_list.append(score)

        return feature_val_list

    def get_freq_term_summary(self):

        freq_terms = set()  # tij >= 0.5 of LS
        for term in self.sentence_frequency_dict.keys():
            if self.sentence_frequency_dict[term] >= math.floor(0.5 * self.summary_length) and term != 'length':
                freq_terms.add(term)
        return freq_terms

    def find_htfs(self):
        max_score = float('-inf')

        for each_sent in self.text:
            if each_sent[0] != '@':
                term_frequency_dict = self.term_frequency(each_sent)  # tfij
                weight = 0
                for word in set(each_sent):
                    inv_doc_score = self.inverse_sentence_frequency(word)
                    weight += term_frequency_dict[word] * inv_doc_score

                max_score = max(max_score, weight)  # HTFS
        return max_score

    def word_sentence_score(self):

        htfs = self.find_htfs()
        freq_terms = self.get_freq_term_summary()
        res = []
        for each_sent in self.text:
            if each_sent[0] != '@':
                term_frequency_dict = self.term_frequency(each_sent)  # tfij
                weight = 0
                for word in set(each_sent):
                    if word in freq_terms:
                        inv_doc_score = self.inverse_sentence_frequency(word)
                        weight += term_frequency_dict[word] * inv_doc_score
                ans = 0.1 + (weight / htfs)
                res.append(ans)

        return res

    def generate_friends(self, sentence, similarity_threshold):
        """
        Args:
            sentence (List[str]): Each sentence in text
            similarity_threshold (float): Similarity threshold to identify a friend
        """
        sentence_string = ' '.join(sentence)
        self.shared_friends_dict[sentence_string] = []
        for individual_sentence in self.text:
            if individual_sentence == sentence or individual_sentence[0] == '@':
                continue
            similarity_score = self.calculate_similarity_sentences(sentence, individual_sentence)
            if similarity_score >= similarity_threshold:
                self.shared_friends_dict[sentence_string].append(' '.join(individual_sentence))
    
    def getfeaturevec(self, doc=None, setngrams=None):
        if doc is not "":
            self.text = doc

        if setngrams is not None:
            self.n_grams = setngrams

        # put your feature functions below
        self.feature_vector_list.append(self.get_Topic_Sentence_Feature())
        # Create sentence frequencies
        self.sentence_frequency()
        for sentence in self.text:
            self.generate_friends(sentence, SIMILARITY_THRESHOLD)
            
        self.feature_vector_list.append(self.sentence_centrality())

        self.feature_vector_list.append(self.word_sentence_score())

        return np.array(self.feature_vector_list).T
