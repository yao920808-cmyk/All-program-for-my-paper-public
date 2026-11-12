# Pathology microscopy tile reconstruction

Reassemble shuffled microscopy tiles using color, stain, and texture at tile boundaries. The pipeline scores neighboring tiles, builds a grid with beam search, and refines the layout with simulated annealing.

## Quick start

From this directory, use a Python environment with CUDA-enabled PyTorch, NumPy, OpenCV, scikit-image, and scikit-learn. Place your images in `image_file/` and replace `example.jpg` with the reconstruction target. Supply additional images for training.

```shell
python pathology_reconstruction_fusion.py --train_dir image_file --test_image image_file/example.jpg --no_existing_slices --rows 4 --cols 5 --output_dir outputs/4x5/fusion
```

Use `--rows 3 --cols 6` or `--rows 5 --cols 5` to change the grid. For pre-cut tiles, replace `--no_existing_slices` with `--slice_root "cutting image"`. Choose a separate output directory for each run.

## Model entry points

| Script | Method |
| --- | --- |
| pathology_reconstruction_random_forest.py | Handcrafted boundary features and random forest |
| pathology_reconstruction_pairnet.py | Learned compatibility between boundary strips |
| pathology_reconstruction_fusion.py | Random forest and PairNet score fusion |
| pathology_reconstruction_graphsage.py | Fusion with GraphSAGE reranking |
| pathology_reconstruction_gat.py | Fusion with graph attention reranking |

## Results

Compare `scrambled.png`, `reconstructed.png`, and `reference_reassembly.png` in the output directory. Read `metrics.json` for adjacency accuracy, absolute position accuracy, structural similarity, and exact reconstruction status.

## Comparison suites

Configure your local paths in `config.py` before running `python run_full_comparison.py`. The launcher runs the traditional classifiers, feature MLP, PairNet, and fusion/GraphSAGE suites with their default arguments. The comparison suites also use pandas and support XGBoost.
