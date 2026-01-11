# TODO: Check against overfitting. When shuffling b, is the manifold a dot?
# TODO: Write unit tests for each function
# TODO: Analyze bifurcation structure of the manifold
# TODO: Rethink behavioural state, maybe fuse some states together
# NOTE: python main.py --downsample_fs 20 --window 50  --batch_size 50 --no_gif --gamma 0.75 --learning_rate 0.0001 --normalize_method minmax --downsample_method gaussian
# TODO: Add comparison noteboko of sownsampling methods and normalization methods
# NOTE: Lazy vs normal loading should be addresses somewhere
# TODO: cahcing of dataste class 

# python main.py --downsample_fs 30 20 25 --window 90 30 50 70  --batch_size 50 --no_gif --gamma 0.5 0.6 0.7 0.8 0.9 --learning_rate 0.0001 0.00001 --normalize_method minmax_global minmax --downsample_method gaussian count