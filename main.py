import gym
import numpy as np
from agent import Agent
# from utils import plot_learning_curve

if __name__ == '__main__':

    env = gym.make('Pendulum-v1')
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0])
    
    n_games = 250
    figure_file = 'plots/pendulum.png'
    best_score = env.reward_range[0]
    score_history = []
    load_chekpoint = False
    
    if load_chekpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation,action,reward,observation_,done)
            n_steps += 1
        agent.learn()
        agent.load_model()
        evaluate = True
    else: evaluate = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        # if 10 & i ==0:
        #     env.render()
            
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score +=reward
            agent.remember(observation,action,reward,observation_,done)
            if not load_chekpoint:
                agent.learn()
            
            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_chekpoint:
                agent.save_model()

        print('episode ', i, 'score %.1f' % score,
               'avg score %.1f' % avg_score)
        
    # if not load_chekpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)

