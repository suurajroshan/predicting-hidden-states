python exp_script.py \
    metric_logger.mode=disabled \
    config_file="configs/llama_3B_PHi_gumbel-quantizer.yaml"  \
    model.self_prediction_module.codebook_dim=10240 \
    model.self_prediction_module.codeword_dim=3072 \
    model.self_prediction_module.num_quantizers=1 \
    model.self_prediction_module.reconstruction_loss_factor=1e-6 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.2 \
    temperature_scheduler.global_steps=10000 \
    optimizer.lr=1e-2 \
    batch_size=4 \
    debug=True \