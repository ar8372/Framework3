import pickle


def save_pickle(path, to_dump):
    with open(path, "wb") as f:
        pickle.dump(to_dump, f)


def load_pickle(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    return o
