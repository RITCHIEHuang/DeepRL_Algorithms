#!/usr/bin/env bash

envs=(MountainCar-v0 CartPole-v1 Acrobot-v1 LunarLander-v2)
seeds=1
version=pytorch
algs=(DQN DoubleDQN DuelingDQN)
for (( k = 0; k < ${#algs[@]}; ++k )); do
    alg=${algs[$k]}
    for (( j = 1; j <= seeds; ++j )); do
        for (( i = 0; i < ${#envs[@]}; ++i )); do
            echo ============================================
            echo Algo: ${alg}, starting Env: ${envs[$i]} ----- Exp_id $j

            python -m Algorithms.${version}.${alg}.main --env_id ${envs[$i]} --model_path Algorithms/${version}/${alg}/trained_models --seed $j --log_path Algorithms/${version}/log/

            echo Algo: ${alg}, finishing Env: ${envs[$i]} ----- Exp_id $j
            echo ============================================
        done
    done
done
