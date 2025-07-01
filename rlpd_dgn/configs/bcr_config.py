from configs import sac_config


def get_config():
    config = sac_config.get_config()


    config.model_cls = "BCRLearner"


    config.bc_loss = 0.1


    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm=True

    return config
