import collections
import argparse

from trpoagent import TRPOAgent

def main(args=None):
    network5()

def network5():
    Variables = collections.namedtuple('params',['device', 'node_select', 'batch_size', 'lamda', 'gamma',
                                                 'step_size', 'seed', 'chances', 'threads', 'obs_dim', 'act_dim',
                                                 'actor_beta', 'actor_eta', 'actor_epochs', 'actor_lr_multiplier',
                                                 'actor_kl_targ', 'actor_policy_logvar', 'actor_clipping_range',
                                                 'actor_hidden_layers', 'actor_hidden_layer_units',
                                                 'critic_epochs', 'critic_hidden_layers', 'critic_hidden_layer_units'])
    params = Variables("/cpu:0", 0.5, 20, 0.995, 0.98,
                       1e-3, 777, 10, 4, 49, 49,
                       1.0, 50, 20, 1.0,
                       0.003, -1.0, None,
                       3, [49, 49, 49],
                       10, 4, [128, 16, 128, 1])

    env_name = "network_5"
    episodes = 100000

    trpo = TRPOAgent(params, env_name, episodes)
    trpo.run()

if __name__ == "__main__":
    main()

