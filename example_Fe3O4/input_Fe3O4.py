import numpy as np
##########
##########BASIC
setting_prefix="Fe3O4_7113_17_21_39_site"#using in class | function
setting_R=5.0
setting_Nsite=24
setting_Nstruct=200
setting_f_element="Fe3O4.ele"
setting_f_cart="Fe3O4.cart"
setting_abc=[8.3941,8.3941,8.3941,90,90,90]
setting_TF_conv=True;
if setting_TF_conv:
    setting_npt=120
else:
    setting_npt=210;#245-35=210
setting_TF_sort_atom=False# 排序相关设置
setting_TF_absorber_IHS=True# 吸收原子是否参与IHS，兼容旧版本数据集
##########GET_DATASET
setting_f_json="Fe3O4_7113_17_21_39.json"
setting_f_par="par168_IHS200_cart-0.3_0.3_scipy.txt"
setting_f_id_train="id200_train.npy"
setting_f_id_test="id200_test.npy"
setting_unconv_begin=35#using in xmu  frop 4 preedge point
setting_conv_xip = np.linspace(7112 - 19, 7112 + 100, 120)#using in conv   interp grid 
##########TRAIN
setting_n_ep=200
setting_n_ep_save=200
setting_cutoff=8.0
# setting_npt=210;#245-35=210
setting_batch_size=32;setting_vt_batch_size=32;
setting_n_train=180*24
setting_n_valid=20*24
##########FITTING
setting_npar=168
setting_opt_lower_bounds=-0.3
setting_opt_upper_bounds=0.3
setting_f_exp="ICSD26410_P1_r5_EF1_Gm17_GH1p25_conv.txt"  #"ICSD26410_P1_r5_EF1_Gm17_GH1p25_conv.txt"  "Fe3O4.nor"
setting_energy_ori=np.linspace(7112-19,7112+100,120)
setting_energy=np.linspace(7112-19,7112+100,120)#energy_es = energy+es_opt
setting_energy_unconv=np.linspace(7107.50,7212.00,210)
setting_weight_xval=7160#split the weight obj
setting_Gamma_hole=1.25
setting_EFermi=7112