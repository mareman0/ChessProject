from recognition import main

def get_fen(path):
    filepath = 'labeled_preprocessed/1B2b3-Kp6-8-8-2k5-8-8-8.png'
    fen = main(filepath)
    return fen;