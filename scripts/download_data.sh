#!/bin/sh
set -e

# get embeddings
mkdir ./data
chmod 777 ./data
mkdir ./data/tmp_folder
mkdir ./data/embeddings
chmod 777 ./data/embeddings
wget -c -O ./data/embeddings/GoogleNews-vectors-negative300.bin.gz https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gunzip ./data/embeddings/GoogleNews-vectors-negative300.bin.gz
python ./scripts/wrap_w2v.py
wget -c -O ./data/embeddings/glove-100 https://www.dropbox.com/s/7y3yxrn0g3gce0m/glove-100?dl=0
wget -c -O ./data/embeddings/knrm.pkl https://www.dropbox.com/s/t68zp97k1il1hmx/knrm.pkl?dl=0
rm ./data/embeddings/GoogleNews-vectors-negative300.bin

# get profession data
mkdir ./data/profession
mkdir ./data/raw_data
chmod 777 ./data/profession
mkdir ./data/profession/sources
wget -c -O ./data/raw_data/ids.txt https://www.dropbox.com/s/k2k3hofnfllp65b/ids.txt?dl=0
wget -c -O ./data/profession/sources/profession_list.txt https://www.dropbox.com/s/yidpy0430v5h36n/profession_list.txt?dl=0
wget -c -O ./data/profession/sources/profession_synonyms.txt https://www.dropbox.com/s/2pht711jx75i02m/profession_synonyms.txt?dl=0
wget -c -O ./data/profession/splits_profession_dictionary.txt https://www.dropbox.com/s/ixhyydb2v0yvf29/splits_profession_dictionary.txt?dl=0

# get hobby data
mkdir ./data/hobby
chmod 777 ./data/hobby
mkdir ./data/hobby/sources
wget -c -O ./data/hobby/sources/hobby_list.txt https://www.dropbox.com/s/9fz8gko1z8gb3av/hobby_list.txt?dl=0
wget -c -O ./data/hobby/sources/hobby_synonyms.txt https://www.dropbox.com/s/colc49a6pdfm37d/hobby_synonyms.txt?dl=0
wget -c -O ./data/hobby/splits_hobby_dictionary.txt https://www.dropbox.com/s/d8xzaxwgbjiir69/splits_hobby_dictionary.txt?dl=0

# get documents
mkdir ./data/documents
wget -c -O ./data/documents/hobby_shr.txt https://www.dropbox.com/s/imx0wv8vfvof506/hobby_wiki-page.txt?dl=0
wget -c -O ./data/documents/hobby_ext.txt https://www.dropbox.com/s/973engxfwyv10pa/hobby_wiki-cat.txt?dl=0
wget -c -O ./data/documents/hobby_pat.txt https://www.dropbox.com/s/9uz5wsmb9bsjbu8/hobby_web-search.txt?dl=0
wget -c -O ./data/documents/profession_shr.txt https://www.dropbox.com/s/rn6q37w0kq58y8z/profession_wiki-page.txt?dl=0
wget -c -O ./data/documents/profession_ext.txt https://www.dropbox.com/s/qfmhh8aeaxyhp7q/profession_wiki-cat.txt?dl=0
wget -c -O ./data/documents/profession_pat.txt https://www.dropbox.com/s/1uqresf25uwpm80/profession_web-search.txt?dl=0