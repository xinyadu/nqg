python preprocess_embedding.py --embedding ../../archive/embeddings/glove.840B.300d.txt \
				--dict data/qg.src.dict \
			    --output data/qg.src.840B.300d.npy

python preprocess_embedding.py --embedding ../../archive/embeddings/glove.840B.300d.txt \
        --dict data/qg.tgt.dict \
        --output data/qg.tgt.840B.300d.npy

python preprocess_embedding.py --embedding ../../archive/embeddings/glove.840B.300d.txt \
        --dict data/qg.par.dict \
        --output data/qg.par.840B.300d.npy
