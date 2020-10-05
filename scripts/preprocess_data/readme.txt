To create data for all charms and baselines, for both seen and unseen experiments run
python preprocess_pipeline.py

# File specification

make_folds.py   -   use binpacking to split attribute values into even folds

create_folds_whitelists.py  -   split the users into folds for unseen exeriment
create_seen_whitelists.py   -   split the users into train and test for seen experiment

input_bert.py   -   create input for charm and bert baselines from user whitelists (input_features representation)
input_indexed.py    -   create input for baselines from user whitelists (vocabulary indexing)
input_plaintext.py  -   create plaintext representation of inputs

docs_for_charm.py   -   create documents for charm from plaintext documents (vocabulary indexing)
docs_for_doublebert.py  -   create documents for bert ir baseline (input_features representation)

