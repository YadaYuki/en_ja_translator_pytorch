from os.path import abspath, dirname, join

PROJECT_ROOT = join(abspath(dirname(__file__)), "..")
CORPUS_PATH = join(PROJECT_ROOT, "corpus", "kftt-data-1.0", "data")

TOK_CORPUS_PATH = join(CORPUS_PATH, "tok")
