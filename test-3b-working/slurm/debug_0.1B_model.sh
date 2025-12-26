python exp_script.py \
    metric_logger.mode=online \
    model.self_prediction_module.reconstruction_loss_factor=0.001 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=30000 \
    batch_size=16 \
    # debug=True \