config_file="configs/llama_3B_PHi.yaml" 

python exp_script.py \
    metric_logger.mode=disabled \
    config_file=$config_file \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    model.latent_loss_factor=1e-4 \
    model.detach_targets=True \
    batch_size=8 \
    debug=True 