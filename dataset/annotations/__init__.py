"""Annotation sub-package for WLASL300.

Contains the annotation build script (:mod:`dataset.annotations.build_annotations`)
that parses ``WLASL_v0.3.json``, verifies video files on disk, generates
stratified splits, and pre-computes Word2Vec embeddings for all 300 classes.

Run the build script before training::

    uv run python dataset/annotations/build_annotations.py \\
        --raw_dir            data/raw \\
        --preprocessing_dir  preprocessing \\
        --wlasl_dir          WLASL300 \\
        --folder2label       folder2label_str.txt \\
        --word2vec_bin       trained_models/embeddings/GoogleNews-vectors-negative300.bin \\
        --out_dir            dataset/annotations
"""
