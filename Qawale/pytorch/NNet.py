import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import time
from NeuralNet import NeuralNet
from Qawale.pytorch import QawaleNNet


class NNetWrapper(NeuralNet):
    """
    Qawale neural network wrapper consistent with the MCTS usage
    and following the style of NNetWrapper in NNet(7).py.
    """
    def __init__(self, game):
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Arguments for model design; can be tuned as needed
        args_cls = lambda: None
        args_cls.num_channels = 64   # example channel size
        args_cls.dropout = 0.3
        args_cls.cuda = torch.cuda.is_available()
        self.args = args_cls

        # Create the actual PyTorch model
        self.nnet = QawaleNNet(game, self.args)
        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        where board is in canonical form,
        pi is a policy distribution over actions,
        and v is the value in [-1,1].
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=1e-3)
        epochs = 10
        batch_size = 64

        for epoch in range(epochs):
            self.nnet.train()
            batch_count = int(len(examples)/batch_size)
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert to tensors
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(target_pis))
                target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float64))

                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # Forward
                out_pi, out_v = self.nnet(boards)

                # Losses
                loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                loss_v = torch.sum((target_vs - out_v.view(-1))**2) / target_vs.size(0)
                total_loss = loss_pi + loss_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with shape (board_x, board_y) in canonical form.
        Returns:
            pi: policy vector (length = action_size)
            v:  float in [-1,1], the board value
        """
        self.nnet.eval()
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        with torch.no_grad():
            pi, v = self.nnet(board)
        # pi is log_softmax, so convert to normal probabilities
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0][0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path " + filepath)
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
