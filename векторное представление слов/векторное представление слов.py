
from itertools import combinations
from pathlib import Path
import gensim

targets = ["картофель_NOUN", "кочерыжка_NOUN"]


model_path = Path("cbow.txt")

model = gensim.models.KeyedVectors.load_word2vec_format(str(model_path), binary=False)

words = []

for query in ([targets[0]], [targets[1]], targets):
    result = model.most_similar(positive=query,  topn=200)

    selected = [
                word for word, score in result
                if word.endswith("_NOUN") and word not in targets
                ][:8]

    for word in selected:
        if word not in words:
            words.append(word)

print("Найденные кандидаты:")
print([word[:-5] for word in words])


if not all(target in model for target in targets):
    raise ValueError("В модели отсутствует одно из заданных слов")

found = None
checks = 0


for a, b in combinations(words, 2):
    for c in words:
        if c in (a, b):
            continue

        checks += 1

        results = model.most_similar(positive=[a, b], negative=[c], topn=100)

        nouns = [
                    (word, score)
                    for word, score in results
                    if word.endswith("_NOUN")
                ][:10]

        if all(
            target in [word for word, _ in nouns]
            for target in targets
        ):
            found = (a, b, c, nouns)
            break

    if found is not None:
        break

print(f"Проверено комбинаций: {checks}")

if found is None:
    print("Подходящая комбинация не найдена")
else:
    a, b, c, nouns = found

    print("Комбинация:", a[:-5], "+", b[:-5], "-", c[:-5])
    print("10 ближайших существительных:")

    for position, (word, score) in enumerate(nouns, 1):
        print(f"{position:2}. {word[:-5]:16} {score:.5f}")

    for target in targets:
        position = next(i for i, (word, _) in enumerate(nouns, 1) if word == target)
        print(f"{target[:-5]}: место {position}")
