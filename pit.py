import Arena
from MCTS import MCTS
import Qawale.QawaleGame
from Qawale.QawaleGame import QawaleGame
from Qawale.QawalePlayers import RandomQawalePlayer, HumanQawalePlayer
from Qawale.pytorch.NNet import NNetWrapper as NNet  # Analog zur Othello-Struktur

import numpy as np
from utils import dotdict

def main():
    human_vs_cpu = True

    # Initialisiere das Qawale-Spiel (4x4 Feld)
    g = QawaleGame()

    # Verschiedene Spieler-Strategien:
    rp = RandomQawalePlayer(g).play  # Zufallsstrategie
    hp = HumanQawalePlayer(g).play   # Menschlicher Spieler

    # KI-Agent (Neural Network), ähnlich wie in Othello
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/qawale/pytorch/', 'qawale_best.pth.tar')

    # MCTS-Parameter (z.B. 50 Simulationen, cpuct=1.0)
    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)

    # Aus MCTS abgeleitete Zugs-Strategie
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    # Entscheide, wer Spieler 2 ist
    if human_vs_cpu:
        player2 = hp
    else:
        # könnte ebenso ein zweites NN sein oder eine andere KI
        player2 = rp

    # Arena zum Austragen der Partien (Display-Funktion optional)
    arena = Arena.Arena(n1p, player2, g, display=QawaleGame.display)

    # Starte beliebig viele Spiele, z.B. 2
    print(arena.playGames(2, verbose=True))

if __name__ == "__main__":
    main()
