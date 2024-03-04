import random

import torch as T
import numpy as np
from torch import tensor, cat, no_grad, mean
import torch.nn.functional as F
import networkx as nx

import os
import datetime
import matplotlib.pyplot as plt

from Agent import Agent
from MultiAgentReplayBuffer import MultiAgentReplayBuffer
from NetworkEngine import NetworkEngine
from NetworkEnv import NetworkEnv
from environmental_variables import STATE_SIZE, EPOCH_SIZE, NUMBER_OF_AGENTS, NR_EPOCHS, EVALUATE, CRITIC_DOMAIN, SIM_NR, NEURAL_NETWORK, MODIFIED_NETWORK, NOTES, TOPOLOGY_TYPE, UPDATE_WEIGHTS, PATH_SIMULATION, GNN_MODULE, GRAPH_BATCH_SIZE, NUMBER_OF_PATHS


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

    def choose_action(self, raw_obs, topology):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], topology)
            actions.append(np.argmax(action))
        return actions

    def learn(self, experience):
        if not experience.ready():
            return
        actor_input, current_state, action_taken, reward_obtained, \
        actor_future_input, future_state, done_flags = experience.sample_buffer()

        processing_device = self.agents[0].actor.device
        current_state_array = np.array(current_state, dtype=np.float32)
        current_state = T.tensor(current_state_array, dtype=T.float).to(processing_device)
        #current_state = T.tensor(current_state, dtype=T.float).to(processing_device)

        action_taken_array = np.array(action_taken, dtype=np.float32)
        action_taken = T.tensor(action_taken_array, dtype=T.float).to(processing_device)

        reward_obtained_array = np.array(reward_obtained, dtype=np.float32)
        reward_obtained = T.tensor(reward_obtained_array, dtype=T.float).to(processing_device)

        future_state_array = np.array(future_state, dtype=np.float32)
        future_state = T.tensor(future_state_array, dtype=T.float).to(processing_device)
        done_flags = T.tensor(done_flags).to(processing_device)

        all_new_actions = []
        previous_actions = []

        for idx, agent in enumerate(self.agents):
            future_actor_input = T.tensor(actor_future_input[idx], dtype=T.float).to(processing_device)

            new_action_policy = agent.target_actor.forward(future_actor_input)

            all_new_actions.append(new_action_policy)
            previous_actions.append(action_taken[idx])

        combined_new_actions = T.cat([act for act in all_new_actions], dim=1)
        combined_old_actions = T.cat([act for act in previous_actions], dim=1)

        for idx, agent in enumerate(self.agents):
            with T.no_grad():
                future_critic_value = agent.target_critic.forward(future_state[idx], combined_new_actions[:,
                                                        idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()

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
    #UPDATE_STEPS = 16
    
    eng = NetworkEngine()
    env = NetworkEnv(eng)

    #n_state = 845
    n_action = NUMBER_OF_PATHS
    
    total_rewards = []
    batch_rewards = []
    agents = eng.get_all_hosts()
    all_hosts = eng.get_all_hosts()

    agent_dim = STATE_SIZE
    agent_dims = [agent_dim for host in all_hosts]

    GAMMA = 0.99
    EXPLORE = 20000
    INITIAL_EPSILON = 0.5
    FINAL_EPSILON = 0.0001
    REPLAY_MEMORY = 50000
    #BATCH = 256

    if CRITIC_DOMAIN == "central_critic":
        critic_dim = len(eng.get_link_usage()) + NUMBER_OF_AGENTS
        critic_dims = [critic_dim for i in range(NUMBER_OF_AGENTS)]
        #critic = eng.get_link_usage()
    elif CRITIC_DOMAIN == "local_critic":
        critic_dim = STATE_SIZE
        #critic = state
        critic_dims = [critic_dim for host in all_hosts]

    maddpg_agents = MADDPG(agent_dims, critic_dims, NUMBER_OF_AGENTS, n_action,
                           fa1=10, fa2=80, fc1=15, fc2=80,
                           alpha=0.0001, beta=0.0001, tau=0.0001,
                           chkpt_dir='.\\tmp\\maddpg\\')

    memory = MultiAgentReplayBuffer(1000, critic_dims, agent_dims, n_action, NUMBER_OF_AGENTS, batch_size=100)
    
    if not EVALUATE:
        nr_epochs = NR_EPOCHS
    else:
        if MODIFIED_NETWORK == "edges":
            nr_epochs = 4
        elif MODIFIED_NETWORK == "intranet":
            nr_epochs = 2

    ## SETUP ##
    #create /home/student/agent_files directory if not found
    path = f'/home/{PATH_SIMULATION}/agent_files{SIM_NR}'
    if not os.path.exists(path):
        print("Creating 'agent_files' directory")
        os.mkdir(path)
    #create /home/student/results directory if not found
    path = f'/home/{PATH_SIMULATION}/results'
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
    if GNN_MODULE:
        path = f'/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_GNN_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}'
    else:
        path = f'/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}'
    os.mkdir(path)

    if not EVALUATE:
        #graph_y_axis = np.zeros(NR_EPOCHS)
        y_axis_training = np.zeros(NR_EPOCHS)
        #graph_x_axis = np.zeros(NR_EPOCHS)
        batch_aux = int(NR_EPOCHS/GRAPH_BATCH_SIZE)
        graph_y_axis = np.zeros(batch_aux)
        graph_x_axis = np.arange(0, batch_aux)
    elif EVALUATE: # and UPDATE_WEIGHTS:
        graph_x_axis = np.zeros(EPOCH_SIZE*4)
        aux = np.zeros(EPOCH_SIZE*4) 
        graph_y_axis = [[0 for _ in range(EPOCH_SIZE*4)] for _ in range(nr_epochs)]
        

    if EVALUATE:
        maddpg_agents.load_checkpoint()

    packet_loss_evaluate = []
    packet_sent_evaluate = []
    experience_pck_lost = 0
    experience_pck_sent = 0

    percentage = np.zeros(nr_epochs)
    percentage_2 = np.zeros(nr_epochs)
    available_bw_epoch = np.zeros(nr_epochs)
    available_bw_epoch_2 = np.zeros(nr_epochs)

    total_package_loss_nr = 0
    total_packets_sent_nr = 0
    total_packets_tried_nr = 0

    for epoch in range(0, nr_epochs):
        total_epoch_reward = []
        total_epoch_pck_loss = 0
        total_epoch_pck_sent = 0

        #print("Epoch: ", epoch)

        if EVALUATE and epoch != 0:
            if MODIFIED_NETWORK == "edges": 
                eng.set_different_topology_edges(epoch)
            elif MODIFIED_NETWORK == "intranet":
                eng.set_different_topology_intranet()

        if not EVALUATE:
            episode_size = EPOCH_SIZE
        else:
            if not UPDATE_WEIGHTS:
                episode_size = EPOCH_SIZE * 4
            else:
                episode_size = EPOCH_SIZE * 4

        available_bw_episode = np.zeros(episode_size)
        available_bw_episode_2 = np.zeros(episode_size)
        
        for e in range(episode_size):
            new_tm = e % 2 == 0
            env.reset(new_tm)

            episode_reward = 0
            total_reward = 0
            total_package_loss = 0
            total_packets_sent = 0
            if EVALUATE:
                total_package_loss_nr = 0
                total_packets_sent_nr = 0
                total_packets_tried_nr = 0
            available_bw_time_steps = np.zeros(100)
            
            for time_steps in range(100):
                actions = {}
                prev_states = {}
                next_dsts = eng.get_nexts_dsts()
                print("\nnext dsts: ", next_dsts)
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

                    states.append(state)

                    if CRITIC_DOMAIN == "central_critic":
                        critic_states.append(np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0))
                    elif CRITIC_DOMAIN == "local_critic":
                        #critic = state.reshape(33)
                        #critic_states = states
                        critic_states.append(state)

                ##tensor of origin - destination communication sequences
                #int_hosts = T.tensor([i+1 for i,n in enumerate(all_hosts)])
                #int_array = T.tensor([int(element) for element in np.array(all_dsts)], dtype=T.int)
                #edge_index = T.stack([int_hosts, int_array], dim=0)
                #print("\n", T.stack([int_hosts, int_array], dim=0))

                ##tensor of edges in network        
                edge_index = T.tensor(list(eng.get_nx_topology().edges), dtype=T.long).t().contiguous()      
                #edge_index = T.tensor(list(eng.get_nx_topology().edges))
                #edge_index = list(eng.get_nx_topology().edges)
                #edge_index = eng.get_nx_topology().edges

                ##adjacency matrix approach
                #edge_index = nx.adjacency_matrix(eng.get_nx_topology())
                
                #edge_index = eng.get_nx_topology()

                actions = maddpg_agents.choose_action(states, edge_index) 

                actions_dict = {}
                for index, host in enumerate(all_hosts):
                    if next_dsts.get(host, ''):
                        prob = -1 if EVALUATE else max(0.1, (0.3 - 0.0001 * epoch))
                        if random.random() < prob:
                            action = random.randint(0, 2)
                        else:
                            action = actions[index]
                        if TOPOLOGY_TYPE == "small_network" or TOPOLOGY_TYPE == "arpanet":
                            if (host in eng.single_con_hosts):
                                action = 0                        #algoritmo tradicional

                        actions_dict[host] = {next_dsts.get(host, ''): action}

                next_states, rewards, done, _ = env.step(actions_dict)

                new_next_states = np.empty((NUMBER_OF_AGENTS, agent_dim), dtype=np.double)

                if CRITIC_DOMAIN == "central_critic":
                    all_critic_new_states = [np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0) for i in
                                         range(NUMBER_OF_AGENTS)]
                elif CRITIC_DOMAIN == "local_critic":
                    #all_critic_new_states = next_states
                    all_critic_new_states = list(next_states.values())
                
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


                available_bw_time_steps[time_steps] = np.average(eng.get_link_usage())

                total_reward += sum(rewards) / NUMBER_OF_AGENTS
                #total_package_loss += eng.statistics['package_loss']
                #total_packets_sent += eng.statistics['package_sent']
                if done:
                    break
            
            available_bw_episode[e] = np.average(available_bw_time_steps)
            available_bw_episode_2[e] = np.average(eng.get_link_usage())
            
            ## DATA
            print(f"episode {e}/{episode_size}, epoch {epoch}/{nr_epochs}")
            #print("Total reward", total_reward)
            #print("Total package loss", ng.statistics['package_loss'])
            #print(" ")

            if (e % 3 == 0 and not EVALUATE) or (EVALUATE and UPDATE_WEIGHTS):
                maddpg_agents.learn(memory)

            total_epoch_reward.append(total_reward)

            #total_package_loss = eng.statistics['package_loss']
            #total_packets_sent = eng.statistics['package_sent']
            
            total_epoch_pck_loss += eng.statistics['package_loss']
            total_epoch_pck_sent += eng.statistics['package_sent']

            total_package_loss_nr = eng.statistics['nr_package_loss']
            total_packets_sent_nr =  eng.statistics['nr_package_sent']
            total_packets_tried_nr = eng.statistics['nr_transmitted_packets']

            experience_pck_lost += total_epoch_pck_loss
            experience_pck_sent += total_epoch_pck_sent
            # print(f"STATISTICS OG {eng.statistics}")

            total_rewards.append(total_reward)
            #batch_rewards.append(total_reward)

            if EVALUATE: #and UPDATE_WEIGHTS:
                graph_y_axis[epoch][e] = int(total_reward)

            # print(f"{'OG' if epoch % 2 == 0 else 'NEW'} REWARD {total_reward}")
            ### episode ends
        
        
        #print(f"total epoch reward {total_epoch_reward}")
        # f.write(f"{epoch} {total_epoch_reward}\n")
        if not EVALUATE:
            #graph_y_axis[epoch] = sum(total_epoch_reward) / len(total_epoch_reward)  #average
            y_axis_training[epoch] = sum(total_epoch_reward) / len(total_epoch_reward) #for saving in the training file
            batch_rewards.append(y_axis_training[epoch]) #save average of epoch
            if ((epoch+1) % GRAPH_BATCH_SIZE) == 0:
                batch_index = int(((epoch+1) / GRAPH_BATCH_SIZE)-1)
                graph_y_axis[batch_index] = sum(batch_rewards)    #/ len(batch_rewards)
                batch_rewards = []

        if epoch % 20 == 0: ##30
            print(f"\n AVERGAE WAS {sum(total_rewards) / len(total_rewards)}")
            total_rewards = []

            if not EVALUATE:
                maddpg_agents.save_checkpoint()
                print("SAVING")

        #saving data while training in data file, so data can be accessed while training
        if not EVALUATE and (epoch+1)%100 == 0:
            #data_file_path = "/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data_while_training.csv"            
            x = np.arange(0, NR_EPOCHS)
            #np.savetxt(data_file_path, (x, y_axis_training), delimiter=',')
            np.savetxt(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data_while_training.csv", (x, y_axis_training), delimiter=',')



        #print(total_epoch_pck_loss)

        if EVALUATE:
            #packet_loss_evaluate[epoch] = total_epoch_pck_loss
            #packet_sent_evaluate[epoch] = total_epoch_pck_sent
            percentage[epoch] = round(((total_epoch_pck_loss/total_epoch_pck_sent)*100), 2)
            percentage_2[epoch] = round(((total_package_loss_nr/total_packets_sent_nr)*100),2)
            available_bw_epoch[epoch] = round(np.average(available_bw_episode),2)
            available_bw_epoch_2[epoch] = round(np.average(available_bw_episode_2), 2)
        ### epoch ends

    ##Data text file
    if GNN_MODULE:
        data_file = open(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_GNN_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.txt", "w")
    else:
        data_file = open(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.txt", "w")
    if EVALUATE:
        if UPDATE_WEIGHTS:
            data_file.write(f"Update Weights\n")
        data_file.write(f"Modified Network: {MODIFIED_NETWORK}\n\n")
        data_file.write(f"Packets lost Original network: {percentage[0]}% \n")
        data_file.write(f"Packets lost Original network (number): {percentage_2[0]}% \n")
        data_file.write(f"Available bandwidth: {available_bw_epoch[0]}% \n")
        data_file.write(f"Available bandwidth 2: {available_bw_epoch[0]}% \n\n")
        for index in range(1, nr_epochs):
            data_file.write(f"Packets lost Modified network ({index}): {percentage[index]}% \n")
            data_file.write(f"Packets lost Modified network (number) ({index}): {percentage_2[index]}% \n")
            data_file.write(f"Available bandwidth ({index}): {available_bw_epoch[index]}% \n")
            data_file.write(f"Available bandwidth 2 ({index}): {available_bw_epoch[index]}% \n\n")
        data_file.write(f"{NOTES}\n")
    else:
        #data_file.write(f"Packets lost when training {round(experience_pck_lost/experience_pck_sent * 100, 2)}% \n")
        data_file.write(f"Packets lost training \"lost_nr/(lost_nr+sent_nr)\" : {round(total_package_loss_nr/(total_package_loss_nr+total_packets_sent_nr) * 100, 2)}% \n")
        data_file.write(f"Packets lost training \"lost_nr/(tried_ nr)\" : {round(total_package_loss_nr/(total_packets_tried_nr) * 100, 2)}% \n")
        data_file.write(f"Packets sent training \"sent_nr/(tried_ nr)\" : {round(total_packets_sent_nr/(total_packets_tried_nr) * 100, 2)}% \n")
        data_file.write(f"\n{NOTES}\n")
    data_file.close    

    ## Build graph
    if not EVALUATE:
        ##only show a few resulta
        #interval = 5
        #interval_data = [value for index, value in enumerate(graph_y_axis) if index % interval == 0]
        #graph_x_axis = np.arange(0, len(interval_data))
        
        x = np.arange(0, NR_EPOCHS)
        
        if CRITIC_DOMAIN == "central_critic":
            plt.title(f"Total reward per epoch - central critic")
        elif CRITIC_DOMAIN == "local_critic":
            plt.title(f"Total reward per epoch - local critic")
        
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.legend()
        plt.plot(graph_x_axis, graph_y_axis, label = {NEURAL_NETWORK})
        #plt.plot(graph_x_axis, interval_data, label = {NEURAL_NETWORK})
        if GNN_MODULE:
            plt.savefig(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_GNN_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.png")
            np.savetxt(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_GNN_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data.csv", (graph_x_axis, graph_y_axis), delimiter=',')
        else:
            plt.savefig(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.png")
            np.savetxt(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data.csv", (graph_x_axis, graph_y_axis), delimiter=',')
            np.savetxt(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data_total.csv", (x, y_axis_training), delimiter=',')

        plt.show()
    elif EVALUATE: # and UPDATE_WEIGHTS:
        graph_x_axis = np.arange(0, episode_size)
 
        plt.plot(graph_y_axis[0], label = "Original network")
        plt.plot(graph_y_axis[1], label = "Scenario 1")
        plt.plot(graph_y_axis[2], label = "Scenario 2")
        plt.plot(graph_y_axis[3], label = "Scenario 3")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.title(f"Rewards - Evaluate")
        
        plt.savefig(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.png")
        #np.savetxt(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data.csv", (graph_x_axis, graph_y_axis), delimiter=',')
        plt.show()
    
