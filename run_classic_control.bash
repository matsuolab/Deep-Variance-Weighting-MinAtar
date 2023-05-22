totaltimesteps=2000000
wandbname=Linear-MDVI-Test
device=cuda


for env in CartPole-v1; do  # LunarLander-v2; do
    for prob in 0.0; do  # 0.1; do
        for seed in 1 2 3; do
            # M-DQN
            poetry run python cleanrl/dqn.py --env-random-prob $prob --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name Weight-Net-M-DQN --weight-type variance-net --device $device
            poetry run python cleanrl/dqn.py --env-random-prob $prob --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name M-DQN --weight-type none --device $device

            # DQN
            poetry run python cleanrl/dqn.py --env-random-prob $prob --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name Weight-Net-M-DQN --weight-type variance-net --kl-coef 0.0 --ent-coef 0.0 --device $device
            poetry run python cleanrl/dqn.py --env-random-prob $prob --total-timesteps $totaltimesteps --env-id $env --track --wandb-project-name $wandbname --seed $seed --exp-name DQN --weight-type none --kl-coef 0.0 --ent-coef 0.0 --device $device 
        done
    done
done