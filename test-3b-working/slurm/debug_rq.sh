python exp_script.py \
    metric_logger.mode=disabled \
    model.self_prediction_module.codebook_dim=1024 \
    model.self_prediction_module.num_quantizers=2 \
    model.self_prediction_module.reconstruction_loss_factor=0.00001 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=5000 \
    batch_size=2 \
    debug=True 