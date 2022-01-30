
def preprocess_text(text):
    for sentence in text:
        for word in sentence:
            if word.isnumeric():
                sentence.remove(word)
            if word == "&":
                sentence.remove(word)
            if word == ".":
                sentence.remove(word)
            elif word == ",":
                sentence.remove(word)
            elif word == "?":
                sentence.remove(word)
            elif word == "!":
                sentence.remove(word)
            elif word == ";":
                sentence.remove(word)
            elif word == "(":
                sentence.remove(word)
            elif word == ")":
                sentence.remove(word)
            elif word == "[":
                sentence.remove(word)
            elif word == "]":
                sentence.remove(word)
    return text

