NR_ACTIVE_CONNECTIONS = 10
ACTIVE_HOST = None
NR_MAX_LINKS = 11
STATE_SIZE = NR_ACTIVE_CONNECTIONS * 2 + NR_MAX_LINKS + 2
#NUMBER_OF_HOSTS = 13
#NUMBER_OF_HOSTS = 25
NUMBER_OF_PATHS = 5
#NUMBER_OF_AGENTS = 25

NR_EPOCHS = 15     #epoch
EPOCH_SIZE = 150    #episode

NOTES = "dueling critic max q value. link bw 1,2,3. reward average, bw=0, disregard single connection host"

### NETWORK TOPOLOGY TYPE ###
#TOPOLOGY_TYPE = "service_provider"
#NUMBER_OF_HOSTS = 65
#NUMBER_OF_AGENTS = 65

TOPOLOGY_TYPE = "small_network"
NUMBER_OF_HOSTS = 25
NUMBER_OF_AGENTS = 25

### TEST COMBINATIONS ###
#1 
CRITIC_DOMAIN = "central_critic"
NEURAL_NETWORK = "duelling_q_network"
#2
#CRITIC_DOMAIN = "central_critic"
#NEURAL_NETWORK = "simple_q_network"
#3
#CRITIC_DOMAIN = "local_critic"
#NEURAL_NETWORK = "duelling_q_network"

### Critic Domain ###
#CRITIC_DOMAIN = "central_critic" #global: eng.get_link_usage()
#CRITIC_DOMAIN = "local_critic"   #local: states
### Neural Network ###
#NEURAL_NETWORK = "duelling_q_network"
#NEURAL_NETWORK = "simple_q_network"

EVALUATE = False
#EVALUATE = True
UPDATE_WEIGHTS = False

### Modified Network In Evaluate  ###
MODIFIED_NETWORK = "bw"
#MODIFIED_NETWORK = "edges"
#MODIFIED_NETWORK = "intranet"
