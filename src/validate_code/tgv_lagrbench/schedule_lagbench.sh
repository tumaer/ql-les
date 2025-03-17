echo "start lagbench validation"
source /home/atoshev/code/lagrangebench/.venv/bin/activate

python /home/tkalinov/code/sph_les_directory/lagrangebench/main.py\
  config=/home/tkalinov/code/sph_les_directory/sph_les/src/validate_code/tgv_lagrbench/gns.yaml \
  gpu=0 