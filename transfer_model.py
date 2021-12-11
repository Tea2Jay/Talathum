# %%
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

# %%
with dnnlib.util.open_url("models\\network-snapshot-001480_gan.pkl") as f:
    gan = legacy.load_network_pkl(f)

# %%
# resume_data
# %%
# G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
# D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
with dnnlib.util.open_url("models\\network-snapshot-000800_cond_gan.pkl") as d:
    cond_gan = legacy.load_network_pkl(d)
# %%
gan['G'].mapping
# %%
cond_gan['G']
# %%
gan['G_ema'].mapping

# %%
cond_gan['G_ema'].mapping.embed.activation
# %%
# G_ema = torch.nn.Module()
# %%
for name in ['G','D','G_ema']:
            misc.copy_params_and_buffers(gan[name], cond_gan[name], require_all=False)
# misc.copy_params_and_buffers(gan['G'],cond_gan['G'],require_all=False)
# %%
cond_gan['G']
# %%
with dnnlib.util.open_url("models\\network-snapshot-000800_cond_gan.pkl") as d:
    cond_gan_new = legacy.load_network_pkl(d)
# %%
with open('models\\frankenstein_model.pkl','wb') as f:
    pickle.dump(cond_gan,f)
# %%
