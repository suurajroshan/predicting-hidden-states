python exp_script.py \
    config_file="configs/llama_0.1B_PHi_gumbel-quantizer.yaml" \
    metric_logger.mode=disabled \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    model.self_prediction_module.latent_loss_factor=1e-8 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.2 \
    temperature_scheduler.global_steps=10000 \
    batch_size=4 \
    debug=True 