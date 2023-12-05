import random

import torch as T
import numpy as np
from torch import tensor, cat, no_grad, mean
import torch.nn.functional as F

import os
import datetime
import matplotlib.pyplot as plt

from Agent import Agent
from MultiAgentReplayBuffer import MultiAgentReplayBuffer
from NetworkEngine import NetworkEngine
from NetworkEnv import NetworkEnv
from environmental_variables import STATE_SIZE, EPOCH_SIZE, NUMBER_OF_AGENTS, NR_EPOCHS, EVALUATE, CRITIC_DOMAIN, NEURAL_NETWORK, MODIFIED_NETWORK, NOTES, TOPOLOGY_TYPE


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, fa1=64, fa2=64, gamma=0.99, tau=0.001, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims[agent_idx],
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir, fc1=fc1, fc2=fc2))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(np.argmax(action))
        return actions

    def learn(self, experience):
        if not experience.ready():
            return
        actor_input, current_state, action_taken, reward_obtained, \
        actor_future_input, future_state, done_flags = experience.sample_buffer()

        processing_device = self.agents[0].actor.device
        current_state = T.tensor(current_state, dtype=T.float).to(processing_device)
        action_taken = T.tensor(action_taken, dtype=T.float).to(processing_device)
        reward_obtained = T.tensor(reward_obtained, dtype=T.float).to(processing_device)
        future_state = T.tensor(future_state, dtype=T.float).to(processing_device)
        done_flags = T.tensor(done_flags).to(processing_device)

        all_new_actions = []
        previous_actions = []

        for idx, agent in enumerate(self.agents):
            future_actor_input = T.tensor(actor_future_input[idx],
                                          dtype=T.float).to(processing_device)

            new_action_policy = agent.target_actor.forward(future_actor_input)

            all_new_actions.append(new_action_policy)
            previous_actions.append(action_taken[idx])

        combined_new_actions = T.cat([act for act in all_new_actions], dim=1)
        combined_old_actions = T.cat([act for act in previous_actions], dim=1)

        for idx, agent in enumerate(self.agents):
            with T.no_grad():
                future_critic_value = agent.target_critic.forward(future_state[idx], combined_new_actions[:,
                                                        idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()

                #print("\n ", reward_obtained[:, idx].shape())
                asd = (1 - done_flags[:, 0].int()) * agent.gamma * future_critic_value
                expected_value = reward_obtained[:, idx] + (1 - done_flags[:, 0].int()) * agent.gamma * future_critic_value

            present_critic_value = agent.critic.forward(current_state[idx], combined_old_actions[:,
                                                        idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()

            critic_loss = F.mse_loss(expected_value, present_critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            current_actor_input = T.tensor(actor_input[idx], dtype=T.float).to(processing_device)
            combined_old_actions_clone = combined_old_actions.clone()
            combined_old_actions_clone[:,
            idx * self.n_actions:idx * self.n_actions + self.n_actions] = agent.actor.forward(current_actor_input)
            actor_loss = -T.mean(agent.critic.forward(current_state[idx], combined_old_actions_clone[:,
                                                      idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

        for agent in self.agents:
            agent.update_network_parameters()


if __name__ == '__main__':
    #row = 9324
    #col = 1
    #UPDATE_STEPS = 16
    
    eng = NetworkEngine()
    env = NetworkEnv(eng)

    #n_state = 845
    n_action = 5 ##3
    # onlineQNetwork = QNetwork()
    
    total_rewards = []
    agents = eng.get_all_hosts()
    all_hosts = eng.get_all_hosts()

    agent_dim = STATE_SIZE
    agent_dims = [STATE_SIZE for host in all_hosts]

    GAMMA = 0.99
    EXPLORE = 20000
    INITIAL_EPSILON = 0.5
    FINAL_EPSILON = 0.0001
    REPLAY_MEMORY = 50000
    BATCH = 256

    if CRITIC_DOMAIN == "central_critic":
        critic_dim = len(eng.get_link_usage()) + NUMBER_OF_AGENTS
        critic_dims = [critic_dim for i in range(NUMBER_OF_AGENTS)]
        #critic = eng.get_link_usage()
    elif CRITIC_DOMAIN == "local_critic":
        #critic_dim = STATE_SIZE + NUMBER_OF_AGENTS
        #critic = state
        critic_dims = [STATE_SIZE for host in all_hosts]

    #critic_dims = [critic_dim for i in range(NUMBER_OF_AGENTS)]

    maddpg_agents = MADDPG(agent_dims, critic_dims, NUMBER_OF_AGENTS, n_action,
                           fa1=10, fa2=80, fc1=15, fc2=80,
                           alpha=0.0001, beta=0.0001, tau=0.0001,
                           chkpt_dir='.\\tmp\\maddpg\\')

    memory = MultiAgentReplayBuffer(1000, critic_dims, agent_dims,
                                    n_action, NUMBER_OF_AGENTS, batch_size=100)
    

    ## SETUP ##
    #create /home/student/agent_files directory if not found
    path = '/home/student/agent_files'
    if not os.path.exists(path):
        print("Creating 'agent_files' directory")
        os.mkdir(path)
    #create /home/student/results directory if not found
    path = '/home/student/results'
    if not os.path.exists(path):
        print("Creating 'results' directory")
        os.mkdir(path)
    #create folder for current simulation
    day = datetime.date.today().day
    month = datetime.date.today().month
    hh = datetime.datetime.now().hour
    mm = datetime.datetime.now().minute
    if EVALUATE:
        learning = "test"
    else:
        learning = "train"
    path = f'/home/student/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}'
    os.mkdir(path)

    graph_y_axis = np.zeros(NR_EPOCHS)
    graph_x_axis = np.zeros(NR_EPOCHS)

    if EVALUATE:
        maddpg_agents.load_checkpoint()

    packet_loss_evaluate = []
    packet_sent_evaluate = []
    experience_pck_lost = 0
    experience_pck_sent = 0

    #nr_trains = 1

    #nr_epochs = NR_EPOCHS if not evaluate else 4
    if not EVALUATE:
        nr_epochs = NR_EPOCHS
    else:
        if MODIFIED_NETWORK == "bw":
            nr_epochs = 4
        elif MODIFIED_NETWORK == "edges":
            nr_epochs = 2
        elif MODIFIED_NETWORK == "intranet":
            nr_epochs = 2

    percentage = np.zeros(nr_epochs)
    available_bw_epoch = np.zeros(nr_epochs)

    for epoch in range(0, nr_epochs):
        total_epoch_reward = 0
        total_epoch_pck_loss = 0
        total_epoch_pck_sent = 0
        #print("Epoch: ", epoch)

        if EVALUATE and epoch != 0:
            if MODIFIED_NETWORK == "bw": #and EVALUATE and epoch != 0:
                eng.set_different_topology_bw(epoch)
            elif MODIFIED_NETWORK == "edges": #and EVALUATE and epoch != 0:
                eng.set_different_topology_edges()
            elif MODIFIED_NETWORK == "intranet": #and EVALUATE and epoch != 0:
                eng.set_different_topology_intranet()

        episode_size = EPOCH_SIZE if not EVALUATE else EPOCH_SIZE * 2
        available_bw_episode = np.zeros(episode_size)
        
        for e in range(episode_size):
            new_tm = e % 2 == 0
            env.reset(new_tm)

            episode_reward = 0
            total_reward = 0
            total_package_loss = 0
            total_packets_sent = 0
            available_bw_time_steps = np.zeros(100)
            
            for time_steps in range(100):
                actions = {}
                prev_states = {}
                next_dsts = eng.get_nexts_dsts()
                #print("next dsts", next_dsts)
                all_dsts = []
                for host in all_hosts:
                    if host in next_dsts and next_dsts[host]:
                        d = next_dsts[host][1:]
                        all_dsts.append(d)
                    else:
                        all_dsts.append(0)

                states = []  # np.empty((50, agent_dim), dtype=np.double)
                critic_states = []
                dismiss_indexes = []

                for index, host in enumerate(all_hosts):
                    all_dst_states = eng.get_state(host, 1)
                    dst = next_dsts.get(host, '')
                    dst = '' if dst == None else dst

                    if 'H' not in dst:
                        state = np.zeros((1, agent_dims[index]), dtype=np.double)
                        dismiss_indexes.append(index)
                    else:
                        state = all_dst_states
                        base_state = all_dst_states

                    #print("\n state: ", state)
                    states.append(state)
                    #print("\n states: ", states)

                    if CRITIC_DOMAIN == "central_critic":
                        critic = eng.get_link_usage()
                        #print("\n (central critic) link usage: ", critic)
                        #print("\n (destino atual) all dsts: ",np.array(all_dsts))
                        critic_states.append(np.concatenate((critic, np.array(all_dsts)), axis=0))
                    elif CRITIC_DOMAIN == "local_critic":
                        #critic = state.reshape(33)
                        #critic_states.append(np.concatenate((critic, np.array(all_dsts)), axis=0))
                        critic_states = states
                
                    #print("\n\n", state)
                    #print("\n shape state ", np.shape(state))
                    #print(state.reshape(33))

                    #critic_states.append(np.concatenate((critic, np.array(all_dsts)), axis=0))

                actions = maddpg_agents.choose_action(states)
                #print("\n actions: ", actions)

                actions_dict = {}
                for index, host in enumerate(all_hosts):
                    if next_dsts.get(host, ''):

                        prob = -1 if EVALUATE else max(0.1, (0.3 - 0.0001 * epoch))

                        if random.random() < prob:
                            action = random.randint(0, 2)
                        else:
                            action = actions[index]

                        if host in eng.single_con_hosts:
                            action = 0                        #algoritmo tradicional

                        actions_dict[host] = {next_dsts.get(host, ''): action}

                #print("\n actions dict: ", actions_dict)
                next_states, rewards, done, _ = env.step(actions_dict)

                #print("\nstep: ", next_states, rewards, done, _)

                #new_next_states = np.empty((25, agent_dim), dtype=np.double)
                new_next_states = np.empty((NUMBER_OF_AGENTS, agent_dim), dtype=np.double)

                if CRITIC_DOMAIN == "central_critic":
                    all_critic_new_states = [np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0) for i in
                                         range(NUMBER_OF_AGENTS)]
                elif CRITIC_DOMAIN == "local_critic":
                    #critic = state.reshape(33)
                    all_critic_new_states = states

                #all_critic_new_states = [np.concatenate((critic, np.array(all_dsts)), axis=0) for i in
                                         #range(NUMBER_OF_AGENTS)]

                #print("\n all critic new states: ", all_critic_new_states)
                
                new_next_states = []
                for index, host in enumerate(all_hosts):
                    # means it add an action
                    if host in actions_dict and next_dsts[host]:
                        bw_state = next_states[host]
                        new_next_states.append(bw_state)
                    else:
                        new_next_states.append(np.zeros((1, agent_dims[index]), dtype=np.double))

                actions = []

                for host in all_hosts:
                    if host not in actions_dict:
                        actions.append(0)
                    else:
                        actions.append(actions_dict[host][next_dsts[host]])

                memory.store_transition(states, actions, rewards, new_next_states, done, critic_states,
                                        all_critic_new_states)

                learn_steps = 0

                available_bw_time_steps[time_steps] = np.average(eng.get_link_usage())

                #total_reward += sum(rewards) / 25
                total_reward += sum(rewards) / NUMBER_OF_AGENTS
                total_package_loss += eng.statistics['package_loss']
                total_packets_sent += eng.statistics['package_sent']
                if done:
                    break
            

            available_bw_episode[e] = np.average(available_bw_time_steps)
            
            ## DATA
            print(f"episode {e}/{episode_size}, epoch {epoch}/{nr_epochs}")
            #print("Total reward", total_reward)
            #print("Total package loss", total_package_loss)
            print(" ")

            if e % 3 == 0 and not EVALUATE:
                maddpg_agents.learn(memory)

            total_epoch_reward += total_reward
            total_epoch_pck_loss += total_package_loss
            total_epoch_pck_sent += total_packets_sent
            experience_pck_lost += total_epoch_pck_loss
            experience_pck_sent += total_epoch_pck_sent
            # print(f"STATISTICS OG {eng.statistics}")

            total_rewards.append(total_reward)

            # print(f"{'OG' if epoch % 2 == 0 else 'NEW'} REWARD {total_reward}")
            ### episode ends

        print(f"total epoch reward {total_epoch_reward}")
        # f.write(f"{epoch} {total_epoch_reward}\n")
        graph_y_axis[epoch] = total_epoch_reward / episode_size
        ##average
        #graph_y_axis[epoch] = sum(total_rewards) / len(total_rewards) ####

        if epoch % 30 == 0:
            print(f"AVERGAE WAS {sum(total_rewards) / len(total_rewards)}")
            total_rewards = []

            if not EVALUATE:
                maddpg_agents.save_checkpoint()
                print("SAVING")

        print(total_epoch_pck_loss)

        if EVALUATE:
            #packet_loss_evaluate[epoch] = total_epoch_pck_loss
            #packet_sent_evaluate[epoch] = total_epoch_pck_sent
            percentage[epoch] = round(((total_epoch_pck_loss/total_epoch_pck_sent)*100), 2)
            available_bw_epoch[epoch] = round(np.average(available_bw_episode),2)
        ### epoch ends

    ##Data text file
    data_file = open(f"/home/student/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.txt", "w")
    if EVALUATE:
        data_file.write(f"Modified Network: {MODIFIED_NETWORK}\n\n")
        data_file.write(f"Packets lost Original network: {percentage[0]}% \n")
        data_file.write(f"Available bandwidth: {available_bw_epoch[0]}% \n\n")
        for index in range(1, nr_epochs):
            data_file.write(f"Packets lost Modified network ({index}): {percentage[index]}% \n")
            data_file.write(f"Available bandwidth ({index}): {available_bw_epoch[index]}% \n\n")
        data_file.write(f"{NOTES}\n")
    else:
        data_file.write(f"Packets lost when training {round(experience_pck_lost/experience_pck_sent * 100, 2)}% \n")
        data_file.write(f"{NOTES}\n")
    data_file.close    

    ## Build graph
    if not EVALUATE:
        graph_x_axis = np.arange(0, NR_EPOCHS)
        if CRITIC_DOMAIN == "central_critic":
            plt.title(f"Total reward per epoch - central critic")
        elif CRITIC_DOMAIN == "local_critic":
            plt.title(f"Total reward per epoch - local critic")
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.plot(graph_x_axis, graph_y_axis, label = {NEURAL_NETWORK})
        plt.savefig(f"/home/student/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.png")
        np.savetxt(f"/home/student/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data.csv", (graph_x_axis, graph_y_axis), delimiter=',')
        plt.legend()
        plt.show()
    else:
        pass
    
