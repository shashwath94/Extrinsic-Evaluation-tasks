# Extrinsic-Evaluation-tasks

For each task, run `preprocess.py` to load the preprocessed version of the dataset.
To train the model, run `train.py`

A pretrained word embedding text file is needed where every line has a word string followed by a space and the embedding vector.
For example, `acrobat 0.6056159735 -0.1367940009 -0.0936380029 0.8406270146 0.2641879916 0.4209069908 0.0607739985 0.5985950232 -1.1451450586 -0.8666719794 -0.5021889806 0.4398249984 0.9671009779 0.7413169742 -0.0954160020 -1.1526989937 -0.3915260136 -0.1520590037 0.0893440023 -0.2578850091 -0.6204599738 -0.8789629936 0.3581469953 0.5509790182 0.1234730035`

Data for NLI task can be found [here](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

For the sequence labeling tasks(POS, NER and chunking), please refer to [this repo](https://github.com/shashwath94/Sequence-Labeling)
