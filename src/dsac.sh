#!/bin/sh


source /opt/conda/etc/profile.d/conda.sh && conda activate tf_1_14

# nohup sh -c "source activate tf_1_14 " "python dsac.py -i "./../dataset/instance/clipped_dataset/" -m "./../models/planetv3/" --do_plot "True" " > start_docker.log 2>&1 &

# python dsac.py -i "./../dataset/instance/clipped_dataset/" -m "./../models/planetv5/" --do_plot "True"

# nohup sh -c "python dsac.py -i "./../dataset/instance/clipped_dataset/" -m "./../models/planetv6/" --do_plot "True" " > dsac_v6.log 2>&1 &
# nohup sh -c "python dsac.py -i "./../dataset/planet_dsac/" -m "./../models/planet_v7/" --do_plot "True" " > dsac_v7.log 2>&1 &

nohup sh -c "python dsac.py -i "./../dataset/planet_dsac_dataset/" -m "./../models/planet_v8/" --do_plot "True" " > dsac_v8.log 2>&1 &

# convert "./../models/planet_v7/results/*.{png,jpeg}" -quality 100 outfile.pdf
# convert "*.{png,jpeg}" -quality 100 outfile.pdf
#nohup sh -c "echo $CONDA_PREFIX && echo $CONDA_DEFAULT_ENV && python dsac_test.py" > dsac_test.log 2>&1 &

# nohup sh -c 'wget "$0" && wget "$1"' "$url1" "$url2" > /dev/null &

