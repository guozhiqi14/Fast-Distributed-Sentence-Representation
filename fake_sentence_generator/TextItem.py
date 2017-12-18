#########################################################################
# File Name: TextItem.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-6-20 15:01
# Description: The class for reading corpus.
#   We thank Yacine Jernite (NYU) for the help with the original
#   version of the TextItem class and the text data.
#########################################################################
from text_unidecode import unidecode
import codecs


def process_wiki_line(line):
    return line


def words(sent):
    return sent.split()


def sentences(words):
    return ' '.join(words)


class TextItem:
    def __init__(self):
        self.paragraphs = []

    def read_wiki(self, lines):
        self.text_type = "wiki"
        self.name = lines[0]
        self.entities = []
        self.section_names = []
        paragraph_title = "Introduction."
        paragraph = []
        new_title = ""
        for (i, line) in enumerate(lines):
            if line.startswith('='):
                new_title += " --- " + unidecode(line)
                if i + 1 < len(lines) and not lines[i + 1].startswith('='):
                    self.section_names += [paragraph_title]
                    self.paragraphs += [[sen for sen, ents in paragraph]]
                    self.entities += [[(j, ents) for j, (sen, ents) in enumerate(paragraph)]]
                    paragraph_title = new_title
                    new_title = ""
                    paragraph = []
            else:
                if i > 0 and len(line) > 0:
                    paragraph += [sent for sent in process_wiki_line(unidecode(line))]
        if len(new_title) > 0:
            self.section_names += [new_title]
            self.paragraphs += [[sen for sen, ents in paragraph]]
            self.entities += [[(j, ents) for j, (sen, ents) in enumerate(paragraph)]]

    def read_book(self, filename):
        self.text_type = "book"
        line_ct = 0
        f = codecs.open(filename, 'r', encoding='utf8', errors='ignore')
        for line in f:
            self.paragraphs += [[' '.join([word for word in words(sen)])
                                 for sen in sentences(unidecode(line).strip())]]
        f.close()