# Neural Question Generation 

## Overview

Implementation of neural question generation system for reading comprehension tasks. Paragraph-level model and sentence-level model will be made available soon.

If you use our data or code, please cite our paper as follows:  
  > @inproceedings{du2017learning,  
  > &nbsp; &nbsp; title={Learning to Ask: Neural Question Generation for Reading Comprehension},  
  > &nbsp; &nbsp; author={Du, Xinya and Shao, Junru and Cardie, Claire},  
  > &nbsp; &nbsp; booktitle={Association for Computational Linguistics (ACL)},  
  > &nbsp; &nbsp; year={2017}  
  > }  

See the [paper](https://arxiv.org/abs/1705.00106),
>"Learning to Ask: Neural Question Generation for Reading Comprehension"

>Xinya Du, Junru Shao and Claire Cardie

>ACL 2017

## Requirements

[Torch7](https://github.com/torch/torch7)

[tds](https://github.com/torch/tds)

## Paragraph-level model

	cd paragraph


### Preprocessing:

#### Generate src/target dictionary

```
th preprocess.lua -config config-preprocess
```

#### Generate embedding files (.t7)

First replace ```<path to embedding txt file>``` in ```preprocess_embedding.sh``` with real path, then run:


	./preprocess_embedding.sh
	
	mkdir data/embs
	
	cd data 
	
	th convert.lua


### Training:

        cd ..

	th train.lua -config config-train

You can adjust the rnn size for paragraph encoder and sentence encoder by changing ```para_rnn_size``` and ```sent_rnn_size``` respectively.

### Generating:

	th translate.lua -model model/<model file name> -config config-trans
	

## Sentence-level model

	cd sentence


### Preprocessing:

#### Generate src/target dictionary

	th preprocess.lua -config config-preprocess


#### Generate embedding files (.t7)

	python preprocess_embedding.py 
	--embedding  <path to embedding txt file>
	--dict ./data/qg.{src,tgt}.dict 
	--output ./data/qg.{src,tgt}.840B.300d.npy

	th ./data/convert.lua


### Training:


	th train.lua -config config-train



### Generating:


	th translate.lua -model model/<model file name> -config config-trans


## Evaluation

	cd qgevalcap
	./eval.py --out_file <path to output file>


## Sample outputs (interesting!)

**Sentence**: however , the rainforest still managed to thrive during these glacial periods , allowing for the survival and evolution of a broad diversity of species . 

**Human**: did the rainforest managed to thrive during the glacial periods ?

**H&S system**: what allowed for the survival and evolution of a broad diversity of species?

**Ours (sentence)**: why do the birds still grow during glacial periods ?

**Ours (paragraph)**: why did the indigenous specis begin to grow during the glacial period ?

---------------

**Sentence**: as an example , income inequality did fall in the united states during its high school movement from 1910 to 1940 and thereafter .

**Human**: during what time period did income inequality decrease in the united states ?

**H&S system**: where did income inequality do fall during its high school movement from 1910 to 1940 and thereafter as an example ?

**Ours (sentence)**: when did income inequality fall in the us ?

**Ours (paragraph)**: when did high school movement begin ?

---------------

**Sentence**: about 61.1 % of victorians describe themselves as christian .

**Human**: what percentage of victorians are christian ?

**H&S system**: who do about 61. 1 % of victorians describe themselves as?

**Ours (sentence)**: what percent of victorians describe themselves as christian ?

**Ours (paragraph)**: what percent of victorians identify themselves as christian ?

## Acknowledgment

Our implementation is adapted from [OpenNMT](http://opennmt.net). The evaluation scripts are adapted from [coco-caption](https://github.com/tylin/coco-caption) repo.

## License

Code is released under [the MIT license](http://opensource.org/licenses/MIT).
