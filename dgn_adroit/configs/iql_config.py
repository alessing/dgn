from configs import sac_config


def get_config():
    config = sac_config.get_config()


    config.model_cls = "IQL"

    config.expectile = 0.8
    config.temperature = 1.0



    config.num_qs = 2
    config.num_min_qs = 1
    config.critic_layer_norm=True

    return config
