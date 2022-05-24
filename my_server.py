import socket
# import main
from threading import Thread
import json
import pandas as pd

HOST = "192.168.1.5"
PORT = 15000


def recieve_jpg(conn):
    conn.settimeout(0.1)
    fp = open("board.jpg", "wb")
    while True:
        try:
            data = conn.recv(4096)
            if data == b"":
                break
        except socket.timeout:
            break
        fp.write(data)
    fp.close()
    # fen = main.get_fen("board.jpg")
    fen = "rnbqk2r/pp3pbp/3p1np1/2pPP3/5P2/2N5/PP4PP/R1BQKBNR"
    debutes = pd.read_csv("debutes_database.csv", usecols=["eco_num", "name", "moves", "FEN"])
    debute = ""
    s = pd.Series(debutes["FEN"]).str.contains(fen, regex=False)
    i = s[s].index.values
    if len(i) != 0:
        debute = debutes['name'][i[0]]
    data = {"FEN": fen, "name": debute}
    data = json.dumps(data)
    conn.sendall(bytes(data, encoding="utf-8"))
    conn.settimeout(None)


def serve_conn(socket):
    conn, addr = socket.accept()
    with conn:
        print(f"Connected by {addr}")
        recieve_jpg(conn)
        print("Done!")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    # обработка подключений
    while True:
        Thread(serve_conn(s)).start()
