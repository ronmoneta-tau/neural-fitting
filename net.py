from dataclasses import dataclass
import jax, flax, orbax
import jax.numpy as jnp
from flax import linen as nn
# from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp
from etils import epath


class MyMLP(nn.Module):
    hidden_width: int
    output_features: int    
    hidden_layers_num: int
    add_bn: bool = True
    scaled_sigmoid: bool = True
    
    def setup(self):
        # Note: it's only semantically a "conv" - 1,1,1 kernel means it's just a voxelwise MLP
        self.conv1 = nn.Conv(
            features=self.hidden_width, kernel_size=(1, 1, 1), padding="SAME", kernel_init=nn.initializers.xavier_uniform()
        )
        self.bln1 = nn.BatchNorm()
        
        self.internal_convs = [
            nn.Conv(
                features=self.hidden_width, 
                kernel_size=(1, 1, 1), 
                padding="SAME", 
                kernel_init=nn.initializers.xavier_uniform()
            ) for jj in range(self.hidden_layers_num)
        ]
        self.internal_BNs = [nn.BatchNorm() for jj in range(self.hidden_layers_num)]
        
        self.conv_out = nn.Conv(
            features=self.output_features, kernel_size=(1, 1, 1), padding="SAME", kernel_init=nn.initializers.xavier_uniform()
        )
    
    def __call__(self, inputs, train=True):
        x = self.conv1(inputs)
        if self.add_bn:
            x = self.bln1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        
        for hli in range(self.hidden_layers_num):
            x = self.internal_convs[hli](x)
            if self.add_bn:
                x = self.internal_BNs[hli](x, use_running_average=not train)
            x = jax.nn.relu(x)        
        
        x = self.conv_out(x) 
        
        if self.scaled_sigmoid:        
            x = nn.sigmoid(x/10)
        else: 
            x = nn.sigmoid(x)  
            
        return x
        

def get_net(input_shape, hidden_width=128, hidden_layers=1, mrf_len=30, extra_inputs=3, output_features=6):    
    """
        input features: 33 [35] = mrf_len (30) + extra_inputs ( T1 + T2 + B1 + [fss, kss] (for cest given MT))                      
        output features: 6 = fs[s], ks[s], and the optional/experimental b1fix, r2cfix, R2fix
    """
    model = MyMLP(hidden_width=hidden_width, output_features=output_features, hidden_layers_num=hidden_layers)
    
    # -- Initialize the model with dummy input --
    # jax convention: input_shape = (batch_size, height, width, depth, inp_channels)
    input_shape = [1] + list(input_shape) + [mrf_len + extra_inputs]  
    dummy_input = jnp.ones(input_shape)
    params = model.init(jax.random.PRNGKey(0), dummy_input, train=True)

    return model, params


@dataclass
class ModelState:
    apply_fn:callable = None
    params:dict = None
    batch_stats:dict = None
    
    
def state2predictor(model_state, frozen_BN_infer=False):    
    params = {'params': model_state.params, 'batch_stats': model_state.batch_stats}
    if frozen_BN_infer:
        nn_predictor = lambda x: model_state.apply_fn(params, x, train=False, mutable=[]) #, None)  # mutable=[]
    else:
        nn_predictor = lambda x: model_state.apply_fn(params, x, train=True, mutable=['batch_stats'])
    return nn_predictor


def get_ckpt_mngr(folder):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=2,
        cleanup_tmp_directories=True, # overwrite?
        create=True  
    )
    mngr = ocp.CheckpointManager(
        epath.Path(folder),
        {
            'model_state': ocp.PyTreeCheckpointer(),
            'config': ocp.PyTreeCheckpointer()
        },
    options=options)
    
    return mngr

def save_ckpt(model_state, config={'train_cfg':{}, 'net_cfg':{}}, folder='/tmp/mymodel/', step=666):
    
    mngr = get_ckpt_mngr(folder)
    mngr.save(step, {'model_state': model_state, 'config': config})    
    

def load_ckpt(folder='/tmp/mymodel/'): #, **get_net_params):

    mngr = get_ckpt_mngr(folder)
    step = mngr.latest_step()  
    restored = mngr.restore(step)

    model_state_d = restored['model_state']
    config = restored['config']

    # print('config', config)
    get_net_kwargs = config['net_cfg']
    
    nnmodel, nnparams = get_net(**get_net_kwargs)
    
    _nn_predictor = state2predictor(ModelState(
        apply_fn=nnmodel.apply, 
        params=model_state_d['params'],
        batch_stats=model_state_d['batch_stats']
        ))    
    #model_params_d = {'params': model_state_d['params'], 'batch_stats': model_state_d['batch_stats']}
    #_nn_predictor = lambda x: nnmodel.apply(model_params_d, x, train=True, mutable=['batch_stats'])
    
    return _nn_predictor, config



