import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import time

from NeuralNet import NeuralNet

class QawaleNNet(nn.Module):
    """
    Example neural network for Qawale,
    adopting a similar style to OthelloNNet.
    """
    def __init__(self, game, args):
        super(QawaleNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()  # should be (4,4)
        self.action_size = game.getActionSize()           # 16
        self.args = args

        # A simple architecture with a few conv layers, then two heads:
        # 1) policy head -> pi (logits for each action)
        # 2) value head  -> v  (scalar in [-1, 1])
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)

        # Flatten for fully-connected layers
        flat_size = args.num_channels * self.board_x * self.board_y
        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # Policy head
        self.fc_policy = nn.Linear(64, self.action_size)

        # Value head
        self.fc_value = nn.Linear(64, 1)

    def forward(self, s):
        # s is batch of boards: (batch_size, board_x, board_y)
        # reshape for conv2d: (batch_size, 1, board_x, board_y)
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))

        # Flatten
        s = s.view(s.size(0), -1)  # (batch_size, flat_size)

        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        # Policy
        pi = self.fc_policy(s)

        # Value in [-1,1]
        v = torch.tanh(self.fc_value(s))
        return F.log_softmax(pi, dim=1), v
    
    def predict(self, board):
        # Create fixed-size numeric board representation
        numeric_board = np.zeros((self.board_x, self.board_y), dtype=np.float32)
        
        # Convert stone stacks to numeric values
        for r in range(self.board_x):
            for c in range(self.board_y):
                if board[r][c]:  # If stack not empty
                    # Use top stone color as value
                    stone = board[r][c][0]  # Get top stone
                    if stone == 'B':
                        numeric_board[r][c] = 1.0
                    elif stone == 'R':
                        numeric_board[r][c] = -1.0
                    elif stone == 'G':
                        numeric_board[r][c] = 0.5  # Yellow stones
        
        # Convert to tensor and add batch dimension
        board_tensor = torch.FloatTensor(numeric_board)
        if self.args.cuda:
            board_tensor = board_tensor.contiguous().cuda()
        board_tensor = board_tensor.view(1, self.board_x, self.board_y)
        
        # Get predictions
        with torch.no_grad():
            pi, v = self.forward(board_tensor)
        
        # Convert to numpy arrays
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]