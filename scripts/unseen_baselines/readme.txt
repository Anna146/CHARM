Directory for baselines for unseen experiments. First run rake and textrank to produce keywords (queries)
and then run_bm25.py to execute them.

# File specification:

rake.py     -   keyword extraction  from Rose et al. 2010.  Automatic keyword extraction fromindividual  documents
How to run:
python rake.py

textrank.py -   keyword extraction baseline from Rada Mihalcea and Paul Tarau. 2004. Textrank: Bringing order into text.
How to run:
python textrank.py

run_bm25.py -   execute the queries from these methods and fulltext queries
How to run:
python run_bm25.py [--hobby|--profession] [--shr|--ext|--pat]

Arguments:
--shr Wikipedia-page collection
--ext Wikipedia-category collection
--pat Web search collection

--profession
--hobby