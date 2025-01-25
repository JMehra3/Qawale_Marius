import pygame
import sys
sys.path.append('..')
import numpy as np, random
from Game import Game 
from .QawaleLogic import Board

class QawaleGame(Game):
    def __init__(self, width=600, height=600):
        super().__init__()
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Qawale")
        self.clock = pygame.time.Clock()
        self.running = True
        self.logic = Board()
        self.bg_color, self.grid_color, self.highlight_color = (220,220,220),(0,0,0),(0,255,0)
        self.blue, self.red, self.yellow = (0,0,255),(255,0,0),(255,255,0)
        self.cell_size, self.margin_left, self.margin_top = 100, 100, 100
        self.distributing = False
        self.distribution_path = []
        self.selected_stack = None
        self.placed_stone_cell = None

    # -------------------------
    # Implementierung von Game
    # -------------------------
    def getInitBoard(self):
        return [list(row) for row in self.logic.board]

    def getBoardSize(self):
        return (4,4)

    def getActionSize(self):
        return 16

    def getNextState(self, board, player, action):
        new_board = [[stack[:] for stack in row] for row in board]
        r, c = divmod(action, 4)
        if len(new_board[r][c])>0:
            new_board[r][c].insert(0, "B" if player==1 else "R")
        return (new_board, -player)

    def getValidMoves(self, board, player):
        """Ermittle gültige Felder entweder zum Setzen (wenn self.distributing == False)
        oder zum Verteilen (wenn self.distributing == True), basierend auf self.distribution_path."""
        valids = [0]*16
        if not self.distributing:
            # Setzen erlaubt auf Feldern mit >=1 Stein
            for r in range(4):
                for c in range(4):
                    if len(board[r][c])>0:
                        valids[r*4 + c] = 1
        else:
            # Verteilen erlaubt nur auf benachbarte Felder, nicht sofort zurück
            sr, sc = self.distribution_path[-1]
            forbidden = self.distribution_path[-2] if len(self.distribution_path)>=2 else None
            for (rr, cc) in [(sr-1, sc),(sr+1, sc),(sr, sc-1),(sr, sc+1)]:
                if 0<=rr<4 and 0<=cc<4 and (forbidden is None or (rr, cc)!=forbidden):
                    valids[rr*4 + cc] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        backup_board = self.logic.board
        backup_cp = self.logic.current_player
        backup_go = self.logic.game_over
        backup_w = self.logic.winner
        self.logic.board = [[stack[:] for stack in row] for row in board]
        self.logic.current_player = 'B' if player==1 else 'R'
        self.logic.game_over = False
        self.logic.winner = None
        self.logic.check_winner()
        res = 0
        if self.logic.game_over:
            if self.logic.winner=='B': res=1
            elif self.logic.winner=='R': res=-1
            elif self.logic.winner=='Unentschieden': res=1e-4
        self.logic.board = backup_board
        self.logic.current_player = backup_cp
        self.logic.game_over = backup_go
        self.logic.winner = backup_w
        return res

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return str(board)

    # -------------------------
    # Pygame-spezifische Methoden
    # -------------------------
    def run(self):
        while self.running:
            self.clock.tick(30)
            self.handle_events()
            self.update()
            self.draw()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.logic.game_over:
                self.handle_mouse_click(pygame.mouse.get_pos())

    def handle_mouse_click(self, pos):
        gp = self.get_grid_pos_from_mouse(pos)
        if not gp: return
        idx = gp[0]*4 + gp[1]
        valids = self.getValidMoves(self.logic.board, self.logic.current_player)

        if valids[idx] == 1:
            if not self.distributing:
                self.logic.place_stone_on_stack(*gp)
                self.placed_stone_cell = gp
                self.selected_stack = gp
                self.distributing = True
                self.distribution_path = [gp]
            else:
                self.distribution_path.append(gp)
                needed = len(self.logic.board[self.selected_stack[0]][self.selected_stack[1]])
                if len(self.distribution_path)-1 == needed:
                    self.logic.distribute_stack(self.distribution_path)
                    self.distributing = False
                    self.distribution_path.clear()
                    self.placed_stone_cell = None
                    self.logic.check_winner()
                    if not self.logic.game_over:
                        self.logic.switch_player()

    def update(self):
        pass

    def draw(self):
        self.screen.fill(self.bg_color)
        for r in range(5):
            y = self.margin_top + r*self.cell_size
            pygame.draw.line(self.screen, self.grid_color, (self.margin_left,y),
                             (self.margin_left+4*self.cell_size,y),2)
        for c in range(5):
            x = self.margin_left + c*self.cell_size
            pygame.draw.line(self.screen, self.grid_color, (x,self.margin_top),
                             (x,self.margin_top+4*self.cell_size),2)
        for r in range(4):
            for c in range(4):
                stack = self.logic.board[r][c]
                xc = self.margin_left + c*self.cell_size + self.cell_size//2
                yc = self.margin_top + r*self.cell_size + self.cell_size//2
                off=0
                for stone in reversed(stack):
                    pygame.draw.circle(self.screen,(0,0,0),(xc,yc-off),20,2)
                    pygame.draw.circle(self.screen,self.get_color_for_stone(stone),(xc,yc-off),18)
                    off+=10
        if self.placed_stone_cell:
            r,c = self.placed_stone_cell
            rx = self.margin_left + c*self.cell_size
            ry = self.margin_top + r*self.cell_size
            pygame.draw.rect(self.screen, self.blue,(rx+2,ry+2,self.cell_size-4,self.cell_size-4),3)
        font = pygame.font.SysFont(None,28)
        if not self.logic.game_over:
            txt = "Aktueller Spieler: "
            txt += "Blau" if self.logic.current_player=="B" else "Rot"
            self.screen.blit(font.render(txt,True,(0,0,0)),(40,30))
            self.highlight_moves(self.getValidMoves(self.logic.board,self.logic.current_player))
        else:
            msg = "Spielende! "
            w = self.logic.winner
            if w=="B": msg+="Blau hat gewonnen!"
            elif w=="R": msg+="Rot hat gewonnen!"
            elif w=="Unentschieden": msg+="Unentschieden!"
            else: msg+="Kein Sieger."
            self.screen.blit(font.render(msg,True,(0,0,0)),(40,30))
        pygame.display.flip()

    def get_color_for_stone(self, s):
        return self.blue if s=="B" else self.red if s=="R" else self.yellow

    def highlight_moves(self, valids):
        for i,v in enumerate(valids):
            if v==1:
                r,c = divmod(i,4)
                rx = self.margin_left + c*self.cell_size
                ry = self.margin_top + r*self.cell_size
                pygame.draw.rect(self.screen,self.highlight_color,(rx+2,ry+2,self.cell_size-4,self.cell_size-4),3)

    def get_grid_pos_from_mouse(self, pos):
        mx,my = pos
        if (self.margin_left<=mx<self.margin_left+4*self.cell_size and 
            self.margin_top<=my<self.margin_top+4*self.cell_size):
            return ((my-self.margin_top)//self.cell_size,(mx-self.margin_left)//self.cell_size)
        return None
