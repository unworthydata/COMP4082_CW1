sentence = "This is a common interview question"
letterMap: dict = {}

for letter in sentence:
    if not letter.isspace():
        letterMap[letter] = letterMap.get(letter, 0) + 1

max_value = max(letterMap, key=letterMap.get)
print(max_value)
