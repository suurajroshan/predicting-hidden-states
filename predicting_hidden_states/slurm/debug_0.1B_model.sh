python exp_script.py \
    metric_logger.mode=online \
    config_file="configs/llama_0.1B_PHi.yaml" \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    model.latent_loss_factor=0.1 \
    batch_size=8 \
    debug=True \