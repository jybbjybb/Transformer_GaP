
EPOCH=40

cd ../../../../..

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_2_partition_parallel_gap_sp_0.9_ep_5_per_train_no_finetune/finetune_round_1/ '--resume /home/results/multi_round_2_partition_parallel_gap_sp_0.9_ep_5_per_train_no_finetune/1_round/3_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '




