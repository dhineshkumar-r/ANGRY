#   coding=utf-8
import math
import sys

sample_text = "जैसा कि आप देख चुके है, यूरोप में आधुनिक राष्ट्रवाद के साथ ही राष्ट्र-राज्यों का भी उदय हुआ। इससे अपने बारे में लोगों की समझ बदलने लगी। वे कौन हैं, उनकी पहचान किस बात से परिभाषित होती है, यह भावना बदल गई। उनमें राष्ट्र के प्रति लगाव का भाव पैदा होने लगा। नए प्रतीकों और चिन्हों ने, नए गीतों और विचारों ने नए संपर्क स्थापित किए और समुदायों की सीमाओं को दोबारा परिभाषित कर दिया। ज़्यादातर देशों में इस नई राष्ट्रपैय पहचान का निर्माण एक लंबी प्रक्रिया में हुआ। आइए देखें कि हमारे देश में यह चेतना किस तरह पैदा हुई?\
 वियतनाम और दूसरे उपनिवेशों की तरह भारत मैं भी आधुनिक राष्ट्रवाद के उदय को परिघटना उपनिवेशवाद विरोधी आंदोलन के साथ गहरे तौर पर जुड़ी हुई थी। औपनिवेशिक शासकों के ख़िलाफ़ संघर्ष के दौरान लोग आपसी एकता को पहचानने लगे थे। उत्पीड़न और दमन के साझा भाव ने विभिन्न समूहों को एक-दूसरे से बाँध दिया था। लेकिन हर वर्ग और समूह पर उपनिवेशवाद का असर एक जैसा नहीं था। उनके अनुभव भी अलग थे और स्वतंत्रता  के मायने भी भिन्न थे। महात्मा गांधी के नेतृत्व में कांग्रेस ने इन समूहों को इकट्ठा करके एक विशाल आंदोलन खड़ा किया। परंतु इस एकता में \
टकराव के बिंदु भी निहित थे।"
document = [s.strip() for s in sample_text.split(u"।")]

sample_summary = "जैसा कि आप देख चुके है, यूरोप में आधुनिक राष्ट्रवाद के साथ ही राष्ट्र-राज्यों का भी उदय हुआ। नए प्रतीकों और चिन्हों ने, नए गीतों और विचारों ने नए संपर्क स्थापित किए और समुदायों की सीमाओं को दोबारा परिभाषित कर दिया। वियतनाम और दूसरे उपनिवेशों की तरह भारत मैं भी आधुनिक राष्ट्रवाद के उदय को परिघटना उपनिवेशवाद विरोधी आंदोलन के साथ गहरे तौर पर जुड़ी हुई थी। महात्मा गांधी के नेतृत्व में कांग्रेस ने इन समूहों को इकट्ठा करके एक विशाल आंदोलन खड़ा किया।"
summary = [s.strip() for s in sample_summary.split(u"।")]

sent_freq_dict = dict()

def term_frequency(sentence):
    frequency = dict()
    terms = sentence.split()

    for term in terms:
        if term in frequency:
            frequency[term] += 1
        else:
            frequency[term] = 1

    for key in frequency.keys():
        frequency[key] = frequency[key] / len(sentence)
    return frequency


def sentence_frequency(document):

    frequencies = dict()
    frequencies["length"] = len(document)

    for sentence in document:
        words = set(sentence.split())
        for word in words:
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1

    return frequencies


def inverse_sentence_frequency(word, frequency_dict):
    n = frequency_dict["length"]
    numerator = math.log(frequency_dict[word] + 1)
    denominator = math.log(n + 1)

    score = (1 - (numerator / denominator))
    return score


def get_freq_term_summary(document):
    global sent_freq_dict
    freq_terms = set()  # tij >= 0.5 of LS
    for term in sent_freq_dict.keys():
        if sent_freq_dict[term] >= math.floor(0.5 * len(summary)) and term != 'length':
            freq_terms.add(term)
    return freq_terms



def find_htfs(document):
    max_score = float('-inf')
    global sent_freq_dict
    sent_freq_dict = sentence_frequency(document)

    for each_sent in document:
        term_frequency_dict = term_frequency(each_sent) # tfij
        words = set(each_sent.split())
        weight = 0
        for word in words:
            inv_doc_score = inverse_sentence_frequency(word, sent_freq_dict)
            weight += term_frequency_dict[word] * inv_doc_score

        max_score = max(max_score, weight)  # HTFS
    return max_score



def word_sentence_score(document):


    htfs = find_htfs(document)
    freq_terms = get_freq_term_summary(document)
    res = []
    for each_sent in document:
        term_frequency_dict = term_frequency(each_sent)  # tfij
        words = set(each_sent.split())
        weight = 0
        for word in words:
            if word in freq_terms:
                inv_doc_score = inverse_sentence_frequency(word, sent_freq_dict)
                weight += term_frequency_dict[word] * inv_doc_score
        ans = 0.1 + (weight/htfs)
        res.append(ans)

    for i in range(len(document)):
        print(document[i], res[i])
    return res

word_sentence_score(document)
