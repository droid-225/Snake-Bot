import torch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:
    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # controls randomness
        self.gamma = 0.9 # discount rate, smaller than 1, usually .8 or .9
        self.memory = deque(maxlen=MAX_MEMORY) # popleft(), automatically removes old data if MAX_MEMORY is reached
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # head danger check blocks
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # current direction check
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight check
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right check
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left check
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction check
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location check
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]

        return np.array(state, dtype=int) # converts boolean array into binary array

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # if number of tuples in memory is greater than batch size 
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns random list of tuples from memory
        else:
            mini_sample = self.memory # return all tuples in memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # zip function basically combines all fields in a tuple

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves (exploration / exploitation tradeoff)
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float) # converts state into a tensor
            prediction = self.model(state0) # executes foward function in model
            move = torch.argmax(prediction).item() # converts tensor value into an int
            final_move[move] = 1

        return final_move

def train():
    plot_scores = [] # list of scores used to plot later
    plot_mean_scores = [] # list of average scores used to plot later
    total_score = 0
    record = 0 # high score
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        state_old = agent.get_state(game) # get old state
        final_move = agent.get_action(state_old) # get next move based on previous state
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move) 
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done) # train short memory

        agent.remember(state_old, final_move, reward, state_new, done) # remember past data

        if done:
            # train long (replay) memeory and plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score:', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()