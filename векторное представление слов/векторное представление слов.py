
import re
import gensim
from pathlib import Path
from itertools import combinations

targets = ["картофель_NOUN", "кочерыжка_NOUN"]

model_path = Path(__file__).resolve().parent / "cbow.txt"

word2vec = gensim.models.KeyedVectors.load_word2vec_format(str(model_path), binary=False)

for target in targets:
    if target not in word2vec:
        raise ValueError(f"Слово {target} отсутствует в модели")

candidates = [
                "репа", "капуста", "морковь", "свекла", "огурец",
                "брюква", "редька", "кочан", "лук", "тыква"
            ]

words = [
            f"{word}_NOUN"
            for word in candidates
            if f"{word}_NOUN" in word2vec
            and f"{word}_NOUN" not in targets
        ]


# words = []

# for query in ([targets[0]], [targets[1]], targets):
#     result = word2vec.most_similar(positive=query, topn=200)

#     selected = [
#          word for word, score in result
#         if word.endswith("_NOUN") and word not in targets
#     ][:8]

#     for word in selected:
#         if word not in words:
#             words.append(word)

# print("Найденные кандидаты:")
# print([word[:-5] for word in words])
# print("Количество кандидатов:", len(words))


found = None
checks = 0

pat = re.compile(r"(.*)_NOUN$")

for a, b in combinations(words, 2):
    for c in words:
        if c in (a, b):
            continue

        checks += 1

        pos = [a, b]
        neg = [c]

        dist = word2vec.most_similar(positive=pos,negative=neg, topn=100)

        nouns = []

        for word, score in dist:
            e = pat.match(word)

            if e is not None:
                nouns.append((word, score))

            if len(nouns) == 10:
                break

        noun_words = [word for word, score in nouns]

        if all(target in noun_words for target in targets):
            found = (a, b, c, nouns)
            break

        if checks % 100 == 0:
            print(f"Проверено комбинаций: {checks}")

    if found is not None:
        break

print(f"Всего проверено комбинаций: {checks}")

if found is None:
    print("Подходящая комбинация не найдена")
else:
    a, b, c, nouns = found

    print()
    print("Найдена комбинация:")
    print(a[:-5], "+", b[:-5], "-", c[:-5])

    print()
    print("10 ближайших существительных:")

    for position, (word, score) in enumerate(nouns, 1):
        e = pat.match(word)

        if e is not None:
            print(f"{position:2}. {e.group(1):16} {score:.5f}")

    print()
    for target in targets:
        position = next(i for i, (word, _) in enumerate(nouns, 1)
                            if word == target
                        )
        print(f"{target[:-5]}: место {position}")
