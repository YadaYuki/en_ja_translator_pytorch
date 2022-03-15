from os.path import abspath, dirname, join

PROJECT_ROOT = join(abspath(dirname(__file__)), "..")
CORPUS_PATH = join(PROJECT_ROOT, "corpus")

KFTT_CORPUS_PATH = join(CORPUS_PATH, "kftt-data-1.0", "data")

KFTT_TOK_CORPUS_PATH = join(KFTT_CORPUS_PATH, "tok")

TANAKA_CORPUS_PATH = join(CORPUS_PATH, "tanaka", "data")


PICKLES_PATH = join(PROJECT_ROOT, "pickles")
NN_MODEL_PICKLES_PATH = join(PICKLES_PATH, "nn")
FIGURE_PATH = join(PROJECT_ROOT, "figure")
