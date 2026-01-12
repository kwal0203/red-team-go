### FActScore
def text_to_sentences(text):
    # transform InstructGPT output into sentences
    sentences = text.split("- ")[1:]
    sentences = [
        sent.strip()[:-1] if sent.strip()[-1] == "\n" else sent.strip()
        for sent in sentences
    ]
    if len(sentences) > 0:
        if sentences[-1][-1] != ".":
            sentences[-1] = sentences[-1] + "."
    else:
        sentences = []
    return sentences
