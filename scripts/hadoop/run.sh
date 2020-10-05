#!/bin/bash
source /home/d5hadoop/new/d5hadoop_env.sh
hadoop fs -rm ids.txt
hadoop fs -put data/raw_data/ids.txt ids.txt

hadoop fs -rm -r raw_data
hadoop jar $CDH_MR2_HOME/hadoop-streaming.jar -file scripts/hadoop/mapper.py -mapper scripts/hadoop/mapper.py -file scripts/hadoop/reducer.py -reducer scripts/hadoop/reducer.py -cacheFile 'ids.txt#ids.txt' -input 'reddit_*/*'  -output raw_data && \
hadoop fs -cat raw_data/part-00000 > data/raw_data/texts.txt && \
python scripts/hadoop/split_attributes.py
echo success


