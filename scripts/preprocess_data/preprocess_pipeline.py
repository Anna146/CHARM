from make_whitelists import *
from input_bert import *
from input_indexed import *
from input_plaintext import *
from docs_for_charm import *
from docs_for_doublebert import *

predicates = ["profession", "hobby"]

for predicate in predicates:
    print("PREPARING DATA FOR", predicate)
    make_whitelists(predicate=predicate)
    # seen branch
    input_bert(predicate=predicate, do_folds=False)
    input_indexed(predicate=predicate)
    input_plaintext(predicate=predicate, do_folds=False)
    # unseen branch
    #split_folds_whitelists(predicate=predicate)
    input_bert(predicate=predicate, do_folds=True)
    input_plaintext(predicate=predicate, do_folds=True)
    # prepare documents
    for coll in ["shr", "ext", "pat"]:
        docs_for_charm(predicate, coll)
        docs_for_doublebert(predicate, coll)