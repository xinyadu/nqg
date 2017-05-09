## Data

This is the data split described in the paper "Learning to Ask: Neural Question Generation for Reading Comprehension." by Du et. al, ACL (2017).

The structure of this folder is:

    data
    ├── processed
    │   ├── src-{train, dev, test}.txt
    │   ├── tgt-{train, dev, test}.txt
    │   └── para-{train, dev, test}.txt
    │  
    ├── raw
    │   ├── train.json
    │   ├── dev.json
    │   └── test.json
    │
    ├── doclist-train.txt
    ├── doclist-dev.txt
    └── doclist-test.txt
   

Our split is done at the article level, the `doclist-*.txt` contains the article titles of each split. We use the original dev set in the SQuAD dataset as our dev set, we split the original training set into our training set and test set.

The `processed` folder includes input sentence files (`src-*.txt`), corresponding questions files (`tgt-*.txt`), and the files of paragraphs which contain the input sentence (`para-*.txt`). The sentences/questions/paragraphs are tokenized.

The `raw` folder includes the raw data files from the SQuAD dataset, split into train, dev, test set.


## Licence

We re-distribute this data split under the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
