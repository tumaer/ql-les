#!/bin/bash

#use source activate_env.sh lagrangebench
#use source activate_env.sh l-les

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {lagrangebench|l-les}"
    exit 1
fi

# Select the environment based on the provided argument
case "$1" in
    lagrangebench)
        source /home/atoshev/code/lagrangebench/.venv/bin/activate
        ;;
    l-les)
        source /home/tkalinov/code/sph_les_directory/sph_les/.venv/bin/activate
        ;;
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 {lagrangebench|l-les}"
        exit 1
        ;;
esac

# Optional: Print which environment is activated
echo "Activated environment: $1"
