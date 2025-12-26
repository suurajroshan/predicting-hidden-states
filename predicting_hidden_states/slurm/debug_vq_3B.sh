python exp_script.py \
    metric_logger.mode=disabled \
    config_file="configs/llama_3B_PHi_vector-quantizer.yaml" \
    model.self_prediction_module.codebook_dim=4096 \
    model.self_prediction_module.codeword_dim=3072 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    batch_size=8 \
    beta_scheduler.saturation_steps=5000 \
    beta_scheduler.beta_max=0.5 \
    debug=True 