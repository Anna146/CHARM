Directory for charm and bert ir models

# How to run models

For bert_knrm.py and bert_bm25.py:
python bert_knrm.py [--hobby|--profession] [--shr|--ext|--pat] [--fold|--seen]

For bert_ir.py:
python bert_ir.py [--hobby|--profession] [--shr|--ext|--pat]

Arguments:
--shr Wikipedia-page collection
--ext Wikipedia-category collection
--pat Web search collection

--seen Seen experiment
--fold Unseen experiment

--profession
--hobby

For example

python bert_knrm.py --ext --fold --hobby
means run CHARM knrm on unseen experiment for hobby using wikipedia-category