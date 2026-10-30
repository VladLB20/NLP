
import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

morph = pymorphy3.MorphAnalyzer()

with open("отрывок из доктор живаго.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences = sent_tokenize(text, language="russian")

for sentence in sentences:

    words = word_tokenize(sentence, language="russian")

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]

        if not word1.isalpha() or not word2.isalpha():
            continue

        p1 = morph.parse(word1)[0]
        p2 = morph.parse(word2)[0]

        if p1.tag.POS not in ("NOUN", "ADJF"):
            continue

        if p2.tag.POS not in ("NOUN", "ADJF"):
            continue

        if p1.tag.number is None or p1.tag.case is None:
            continue

        if p1.tag.number != p2.tag.number:
            continue

        if p1.tag.case != p2.tag.case:
            continue

        if p1.tag.number == "sing":
            if p1.tag.gender is None or p2.tag.gender is None:
                continue

            if p1.tag.gender != p2.tag.gender:
                continue

        print(p1.normal_form, p2.normal_form)
