from shared import *

class pix():
    axis = 'z'
    dataset_name = '.../PM/InputData/2D-z/Model1'
    input_c_dim = 4
    output_c_dim = 3
    speed_factor = 1
    batch_size = 1
    gf_dim = 128
    df_dim = 128
    lr = 0.0002 # 0.0002
    phase = 'test'
    
var = pix()
init(var)
run(var)
