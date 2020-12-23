#!/usr/bin/env bash

#envs=(HalfCheetah-v3 Hopper-v3 Walker2d-v3 Swimmer-v3 Ant-v3 BipedalWalker-v3)
envs=(BipedalWalker-v3)
seeds=10
max_iter=1000
alg=TRPO
for (( j = 1; j <= seeds; ++j )); do
    for (( i = 0; i < ${#envs[@]}; ++i )); do
        echo ============================================
        echo starting Env: ${envs[$i]} ----- Exp_id $j

        python -m Algorithms.pytorch.${alg}.main --env_id ${envs[$i]} --max_iter ${max_iter} --model_path Algorithms/pytorch/${alg}/trained_models --seed $j --num_process 4

        echo finishing Env: ${envs[$i]} ----- Exp_id $j
        echo ============================================
    done
done

