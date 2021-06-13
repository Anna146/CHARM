# CHARM
Use Python 3.6

More clarifications are in readme.txt in each folder. 
Check the scripts before running, as some parameters may need to be changed in the code (not all parameters are passed through command line).

Proceed as follows

## 0. Download data
sh scripts/download_data.sh

## 1. Fetch Reddit texts
The file data/raw_data/ids.txt contains only message ids but not the messages. Get the message texts yourself, the text to be taken is 'selftext' + 'body' + 'title' from api's json. You can either crawl only the post ids we used using the Reddit API, or you can obtain all the reddit data from another source, such as https://files.pushshift.io/reddit/

A Hadoop script for automatically extracting the needed messages and cleaning them is available in scripts/hadoop/. It expects to find reddit_comments and reddit_submission is in the user's hadoop home directory. Then run

sh scripts/hadoop/run.sh

## 2. Create datasets
python scripts/preprocess_data/preprocess_pipeline.py

## 3. Run seen baselines
python scripts/seen_baselines/cluster_baseline.py [--hobby|--profession]

python scripts/seen_baselines/svm_baseline.py [--hobby|--profession]

python scripts/seen_baselines/bert_baseline.py [--hobby|--profession]

## 4. Run unseen baselines
python scripts/unseen_baselines/rake.py

python scripts/unseen_baselines/textrank.py

python scripts/unseen_baselines/run_bm25.py [--hobby|--profession] [--shr|--ext|--pat]

python scripts/models/bert_ir.py [--hobby|--profession] [--shr|--ext|--pat] --device=0

## 5. Run CHARMS
python scripts/models/bert_knrm.py [--hobby|--profession] [--shr|--ext|--pat] [--fold|--seen]

python scripts/models/bert_bm25.py [--hobby|--profession] [--shr|--ext|--pat] [--fold|--seen]

## 6. Calculate metrics
python scripts/process_results/results_seen.py [--hobby|--profession]

python scripts/process_results/results_unseen.py [--hobby|--profession]

The evaluation metrics, which are used in the paper are in scripts/compute_statistics.py file on lines:

Macro MRR - line 106

Micro nDCG - line 83

Arguments:

--shr Wikipedia-page collection

--ext Wikipedia-category collection

--pat Web search collection


--seen Seen experiment

--fold Unseen experiment


--profession

--hobby


--device=0 give the number of GPU

Data directory stucture:

  - bert_documents    -   contains processed documents (indexed representation) for charms
  
  - documents -   text representation of the documents
  
  - double_bert_documents -   documents for bert IR baseline (input_features representation)
  
  - embeddings    -   contains vocabulary embeddings, model checkpoints and other necessary things
  
  - raw_data  -   contains raw reddit posts
  
  - tmp_folder    -   for intermediate files
  
  - profession, hobby -   contain processed datasets (subfolders datasets, seen_baselines_datasets)
                                unseen baselines queries (subfolder queries)
                                results with output files and matrics per fold (subfolder results)
