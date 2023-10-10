NR_ACTIVE_CONNECTIONS = 10
ACTIVE_HOST = None
NR_MAX_LINKS = 11
STATE_SIZE = NR_ACTIVE_CONNECTIONS * 2 + NR_MAX_LINKS + 2
NUMBER_OF_HOSTS = 13
NUMBER_OF_PATHS = 5
NUMBER_OF_AGENTS = 25

NR_EPOCHS = 10      #epoch
EPOCH_SIZE = 25  #episode


### Critic Domain ###
CRITIC_DOMAIN = "central_critic" #global: eng.get_link_usage()
#CRITIC_DOMAIN = "local_critic"   #local: state


EVALUATE = True

### Modified Network In Evaluate  ###
#MODIFIED_NETWORK = 1
#MODIFIED_NETWORK = 2
#MODIFIED_NETWORK = 3