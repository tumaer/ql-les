#nohup bash /home/tkalinov/code/sph_les_directory/sph_les/validate_code/tgv_lagrbench/schedule_lagbench.sh > /home/tkalinov/code/sph_les_directory/sph_les/validate_code/tgv_lagrbench/gns_LB_64.log &
echo "start lagbench validation"
source /home/tkalinov/code/sph_les_directory/lagrangebench/venv/bin/activate

python /home/tkalinov/code/sph_les_directory/lagrangebench/main.py\
  config=/home/tkalinov/code/sph_les_directory/sph_les/validate_code/tgv_lagrbench/gns.yaml \
  gpu=0 