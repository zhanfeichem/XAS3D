import numpy as np
##########
##########BASIC
setting_prefix="Mn03_EF3def_ihs6000_site"#"Mn_mu_site"
setting_R=5.0
setting_Nsite=1 #Number of Absorber
setting_Nstruct=10000
setting_f_element="Mn_cen.ele"
setting_f_cart="Mn_cen_dft.cart"
setting_f_par="par84_IHS10000_cart-0.3_0.3_scipy.txt"#"par81_IHS3000_-0.3_0.3.txt"
setting_abc=[5.5, 5.5, 8.5,90,90,90]
setting_TF_conv=True
if setting_TF_conv:
    setting_npt=239
else:
    setting_npt=205;#245-40
setting_TF_sort_atom=False# 排序相关设置
setting_TF_absorber_IHS=True# 吸收原子是否参与IHS，兼容旧版本数据集False
##########GET_DATASET
setting_f_json="Mn_6542_15_30_30_IHS6000.json"#Mn_6542.3_7.21_38.34_24.63.json"#"Mn_6542.3_7.21_38.34_24.63.json"#"Mn_6539_15_30_30.json"
setting_f_id_train="id6000_train.npy"
setting_f_id_test="id6000_test.npy"
setting_unconv_begin=40# 
setting_TF_ip_conv=False# False using setting_conv_begin;True using setting_conv_xip
setting_conv_begin=6
setting_conv_xip = np.linspace(6539-20+1,6539+100,239)#np.linspace(6539-20+1,6539+100,120)  #239(0.5eV)  477(0.25eV)  1191(0.1eV) 2381(0.05eV) 
##########TRAIN
setting_n_ep=1000
setting_n_ep_save=200
setting_cutoff=5.0
setting_batch_size=32;setting_vt_batch_size=32;
setting_n_train=32*168
setting_n_valid=624
##########FITTING
setting_maxtime=3600*10#3600*10
setting_maxeval=10000*8
setting_npar=84
setting_opt_lower_bounds=-0.3
setting_opt_upper_bounds=0.3
setting_f_exp="JT.exp"
setting_energy_ori=setting_conv_xip#out = np.vstack([energy_ori, mu_pred]);np.savetxt("opt_res_mu_pred.txt", out.T)
setting_energy=setting_conv_xip#energy_es = energy+es_opt
setting_energy_unconv=np.linspace(6537,6639,205)
setting_weight_xval=6592#split the weight obj
setting_Gamma_hole=1.16
setting_EFermi=6539
#Efermi=6539;Gamma_max=15;Ecent=30;Elarg=30;Gamma_hole=1.16;