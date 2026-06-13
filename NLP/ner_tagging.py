import spacy
from spacy import displacy
from spacy import tokenizer
from IPython.display import HTML,display
import re

nlp=spacy.load("en_core_web_sm")

google_text="""
Google has been one of the most innovative companies in the world
with a long history of technological achievements
from its founding in 1998 by Larry Page and Sergey Brin
to its current position as a global leader in artificial intelligence
and cloud computing
"""

spacy_doc=nlp(google_text)
for word in spacy_doc.ents:
    print(word.text,word.label_)

html=displacy.render(spacy_doc,style="ent",jupyter=False)
display(HTML(html))


google_text_clean=re.sub(r'[^[^\w\s]',' ',google_text).lower()
spacy_doc_clean=nlp(google_text_clean)

for word in spacy_doc_clean.ents:
    print(word.text,word.label_)

html=displacy.render(spacy_doc_clean,style="ent",jupyter=False)
display(HTML(html))