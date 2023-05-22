totaltimesteps=10000000
wandbname=minatar
device=cuda

for env in breakout seaquest asterix freeway space_invaders; do
    for seed in {1..5}; do
        # Weighted M-DQN
        poetry run python cleanrl/dqn_minatar.py --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name Weight-Net-M-DQN --weight-type variance-net --device $device
        # M-DQN
        poetry run python cleanrl/dqn_minatar.py --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name M-DQN --weight-type none --device $device 

        # Weighted DQN
        poetry run python cleanrl/dqn_minatar.py --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name Weight-Net-M-DQN --weight-type variance-net --kl-coef 0.0 --ent-coef 0.0 --device $device
        # DQN
        poetry run python cleanrl/dqn_minatar.py --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name DQN --weight-type none --kl-coef 0.0 --ent-coef 0.0 --device $device
    done
done
