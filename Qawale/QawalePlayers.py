# QawalePlayers.py

import random

class RandomQawalePlayer:
    """
    Ein Beispiel-Zufallsspieler für Qawale.
    """
    def __init__(self):
        pass

    def play(self, game_state):
        """
        game_state könnte enthalten:
            - board: aktuelles Brett
            - valid_moves: Liste aller möglichen Züge
        """
        possible_moves = game_state.get("valid_moves", [])
        if not possible_moves:
            return None
        return random.choice(possible_moves)

class HumanQawalePlayer:
    """
    Ein menschlicher Spieler (hier nur Konzept),
    könnte z.B. Eingaben über die Konsole oder Maus abfragen.
    """
    def __init__(self):
        pass

    def play(self, game_state):
        possible_moves = game_state.get("valid_moves", [])
        if not possible_moves:
            return None
        print("Mögliche Züge:", possible_moves)
        usr_inp = input("Wähle Zug (z.B. '2 1') -> ")
        try:
            r, c = map(int, usr_inp.split())
            if (r, c) in possible_moves:
                return (r, c)
        except ValueError:
            pass
        return None

class GreedyQawalePlayer:
    """
    Ein einfacher "gieriger" Bot – hier nur Zufall als Platzhalter.
    """
    def __init__(self):
        pass

    def play(self, game_state):
        possible_moves = game_state.get("valid_moves", [])
        if not possible_moves:
            return None
        # Aktuell zufällig - hier könnte man eine Heuristik einbauen
        return random.choice(possible_moves)
