_base_ = '../default.py'

expname = 'dvgo_zju_313'
basedir = './logs/zju_mocap'

data = dict(
    datadir='./data/TanksAndTemple/Barn',
    dataset_type='zju_mocap',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
    movie_render_kwargs={'flip_up_vec': True},
)

coarse_train = dict(
    pervoxel_lr_downrate=2,
)

fine_train = dict(pg_scale=[1000,2000,3000,4000,5000,6000])
fine_model_and_render = dict(num_voxels=256**3)

