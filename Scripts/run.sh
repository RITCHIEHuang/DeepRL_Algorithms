#!/usr/bin/env bash

# default setting
VERSION=pytorch
SEED=1
ALGORITHM=DQN
ENVIRONMENT=MountainCar-v0

TEMP=`getopt -o hv:e:s:a: --long help,version:,environment:,seed:,algorithm: \
             -n 'DeepRL_Algorithms Runner' -- "$@"`

if [ $? != 0 ] ; then
    echo "Terminating ..." >& 2 ;
    exit 1;
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -v | --version )
            VERSION=$2
            shift 2
            ;;
        -e | --environment )
            ENVIRONMENT=$2
            shift 2
            ;;
        -s | --seed )
            SEED=$2
            shift 2
            ;;
        -a | --algorithm )
            ALGORITHM=$2
            shift 2
            ;;
        -h | --help )
            echo "Usage: $0 [-v|--version[=]<version>] [-e|--environment[=]<environment>] [-a|--algorithm[=]<algorithm>] \
                            [-s|--seed[=]<seed>"
            echo "v | version version : use \`version\` to run corresponding implementation , \`version\` can be \`tf2\` or \`pytorch\`"
            echo "e | environment environment : specify environment id as \`environment\`"
            echo "a | algorithm : specify algorithm name as \`algorithm\`"
            echo "s | seed : specify random seed as \`seed\`"

            exit 2
            ;;
        -- )
            shift
            break
            ;;
        * )
            break
            ;;
    esac
done

echo ============================================
echo Starting ${ALGORITHM}/${VERSION} on environment: ${ENVIRONMENT} with seed: ${SEED}

python -m Algorithms.${VERSION}.${ALGORITHM}.main --env_id ${ENVIRONMENT} \
                                                  --model_path Algorithms/${VERSION}/${ALGORITHM}/trained_models \
                                                  --seed ${SEED}  \
                                                  --log_path Algorithm/${VERSION}/log/

echo Finishing ${ALGORITHM}/${VERSION} on environment: ${ENVIRONMENT} with seed: ${SEED}
echo ============================================


