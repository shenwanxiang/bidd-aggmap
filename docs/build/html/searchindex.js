Search.setIndex({docnames:[".ipynb_checkpoints/aggmap-checkpoint",".ipynb_checkpoints/api-checkpoint",".ipynb_checkpoints/how_aggmap_works-checkpoint",".ipynb_checkpoints/index-checkpoint",".ipynb_checkpoints/modules-checkpoint",".ipynb_checkpoints/performance-checkpoint","aggmap","aggmap.aggmodel","aggmap.utils","api","auto_examples/.ipynb_checkpoints/index-checkpoint","auto_examples/index","how_aggmap_works","index","modules","performance"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:[".ipynb_checkpoints/aggmap-checkpoint.rst",".ipynb_checkpoints/api-checkpoint.rst",".ipynb_checkpoints/how_aggmap_works-checkpoint.rst",".ipynb_checkpoints/index-checkpoint.rst",".ipynb_checkpoints/modules-checkpoint.rst",".ipynb_checkpoints/performance-checkpoint.rst","aggmap.rst","aggmap.aggmodel.rst","aggmap.utils.rst","api.rst","auto_examples/.ipynb_checkpoints/index-checkpoint.rst","auto_examples/index.rst","how_aggmap_works.rst","index.rst","modules.rst","performance.rst"],objects:{"":[[6,0,0,"-","aggmap"]],"aggmap.AggMapNet":[[9,1,1,"","MultiClassEstimator"],[9,1,1,"","MultiLabelEstimator"],[9,1,1,"","RegressionEstimator"],[6,4,1,"","clean"],[9,4,1,"","load_model"],[6,4,1,"","save_model"]],"aggmap.AggMapNet.MultiClassEstimator":[[6,2,1,"","clean"],[9,3,1,"","explain_model"],[6,3,1,"","fit"],[9,3,1,"","get_params"],[6,3,1,"","load_model"],[6,3,1,"","plot_model"],[6,3,1,"","predict"],[9,3,1,"","predict_proba"],[6,3,1,"","save_model"],[9,3,1,"","score"],[9,3,1,"","set_params"]],"aggmap.AggMapNet.MultiLabelEstimator":[[6,2,1,"","clean"],[9,3,1,"","explain_model"],[6,3,1,"","fit"],[9,3,1,"","get_params"],[6,3,1,"","load_model"],[6,3,1,"","plot_model"],[6,3,1,"","predict"],[9,3,1,"","predict_proba"],[6,3,1,"","save_model"],[9,3,1,"","score"],[9,3,1,"","set_params"]],"aggmap.AggMapNet.RegressionEstimator":[[6,2,1,"","clean"],[9,3,1,"","explain_model"],[6,3,1,"","fit"],[9,3,1,"","get_params"],[6,3,1,"","load_model"],[6,3,1,"","plot_model"],[9,3,1,"","predict"],[6,3,1,"","save_model"],[9,3,1,"","score"],[9,3,1,"","set_params"]],"aggmap.aggmodel":[[7,0,0,"-","cbks"],[7,0,0,"-","explain_dev"],[9,0,0,"-","explainer"],[7,0,0,"-","loss"],[7,0,0,"-","net"]],"aggmap.aggmodel.cbks":[[7,1,1,"","CLA_EarlyStoppingAndPerformance"],[7,1,1,"","Reg_EarlyStoppingAndPerformance"],[7,4,1,"","prc_auc_score"],[7,4,1,"","r2_score"]],"aggmap.aggmodel.cbks.CLA_EarlyStoppingAndPerformance":[[7,3,1,"","evaluate"],[7,3,1,"","on_epoch_end"],[7,3,1,"","on_train_begin"],[7,3,1,"","on_train_end"],[7,3,1,"","roc_auc"],[7,3,1,"","sigmoid"]],"aggmap.aggmodel.cbks.Reg_EarlyStoppingAndPerformance":[[7,3,1,"","evaluate"],[7,3,1,"","on_epoch_end"],[7,3,1,"","on_train_begin"],[7,3,1,"","on_train_end"],[7,3,1,"","r2"],[7,3,1,"","rmse"]],"aggmap.aggmodel.explain_dev":[[7,4,1,"","GlobalIMP"],[7,4,1,"","LocalIMP"],[7,4,1,"","islice"]],"aggmap.aggmodel.explainer":[[9,1,1,"","shapley_explainer"],[7,1,1,"","simply_explainer"]],"aggmap.aggmodel.explainer.shapley_explainer":[[7,3,1,"","covert_mpX_to_shapely_df"],[7,3,1,"","get_shap_values"],[9,3,1,"","global_explain"],[9,3,1,"","local_explain"]],"aggmap.aggmodel.explainer.simply_explainer":[[7,3,1,"","global_explain"],[7,3,1,"","local_explain"]],"aggmap.aggmodel.loss":[[7,4,1,"","MALE"],[7,4,1,"","cross_entropy"],[7,4,1,"","weighted_cross_entropy"]],"aggmap.aggmodel.net":[[7,4,1,"","Inception"],[7,4,1,"","count_trainable_params"],[7,4,1,"","resnet_block"]],"aggmap.map":[[9,1,1,"","AggMap"],[6,1,1,"","Base"],[6,1,1,"","Random_2DEmbedding"]],"aggmap.map.AggMap":[[9,3,1,"","batch_transform"],[6,3,1,"","copy"],[9,3,1,"","fit"],[6,3,1,"","load"],[6,3,1,"","plot_grid"],[9,3,1,"","plot_scatter"],[6,3,1,"","plot_tree"],[9,3,1,"","refit_c"],[6,3,1,"","save"],[9,3,1,"","to_nwk_tree"],[9,3,1,"","transform"],[9,3,1,"","transform_mpX_to_df"]],"aggmap.map.Base":[[6,3,1,"","MinMaxScaleClip"],[6,3,1,"","StandardScaler"]],"aggmap.map.Random_2DEmbedding":[[6,3,1,"","fit"]],"aggmap.show":[[6,4,1,"","imshow"],[6,4,1,"","imshow_wrap"]],"aggmap.utils":[[8,0,0,"-","calculator"],[8,0,0,"-","distances"],[8,0,0,"-","gen_nwk"],[8,0,0,"-","logtools"],[8,0,0,"-","matrixopt"],[8,0,0,"-","multiproc"],[8,0,0,"-","summary"],[9,0,0,"-","vismap"]],"aggmap.utils.calculator":[[8,4,1,"","pairwise_distance"]],"aggmap.utils.distances":[[8,4,1,"","GenNamedDist"],[8,4,1,"","bray_curtis"],[8,4,1,"","canberra"],[8,4,1,"","chebyshev"],[8,4,1,"","correlation"],[8,4,1,"","cosine"],[8,4,1,"","dice"],[8,4,1,"","euclidean"],[8,4,1,"","hamming"],[8,4,1,"","jaccard"],[8,4,1,"","kulsinski"],[8,4,1,"","manhattan"],[8,4,1,"","rogers_tanimoto"],[8,4,1,"","sokal_sneath"],[8,4,1,"","sqeuclidean"]],"aggmap.utils.gen_nwk":[[8,4,1,"","dfs_to_tree"],[8,4,1,"","dfs_to_weightless_newick"],[8,4,1,"","mp2newick"],[8,4,1,"","pprint_tree"],[8,4,1,"","tree"],[8,4,1,"","tree_add"],[8,4,1,"","tree_to_newick"]],"aggmap.utils.logtools":[[8,1,1,"","PBarHandler"],[8,4,1,"","clip_text"],[8,4,1,"","create_print_method"],[8,4,1,"","format_exc"],[8,4,1,"","get_date"],[8,4,1,"","get_datetime"],[8,4,1,"","log_to_file"],[8,4,1,"","pbar"],[8,4,1,"","print_debug"],[8,4,1,"","print_error"],[8,4,1,"","print_exc"],[8,4,1,"","print_exc_s"],[8,4,1,"","print_info"],[8,4,1,"","print_timedelta"],[8,4,1,"","print_traceback"],[8,4,1,"","print_warn"],[8,4,1,"","reset_handler"],[8,4,1,"","set_level"],[8,4,1,"","set_text_length"]],"aggmap.utils.logtools.PBarHandler":[[8,3,1,"","emit"]],"aggmap.utils.matrixopt":[[8,1,1,"","Scatter2Array"],[8,1,1,"","Scatter2Grid"],[8,4,1,"","conv2"],[8,4,1,"","fspecial_gauss"],[8,4,1,"","smartpadding"]],"aggmap.utils.matrixopt.Scatter2Array":[[8,3,1,"","fit"],[8,3,1,"","transform"]],"aggmap.utils.matrixopt.Scatter2Grid":[[8,3,1,"","fit"],[8,3,1,"","refit_c"],[8,3,1,"","transform"]],"aggmap.utils.multiproc":[[8,4,1,"","ImapUnorder"],[8,4,1,"","MultiExecutorRun"],[8,4,1,"","MultiProcessRun"],[8,4,1,"","MultiProcessUnorderedBarRun"],[8,4,1,"","RunCmd"]],"aggmap.utils.summary":[[8,1,1,"","Summary"],[8,4,1,"","Summary2"]],"aggmap.utils.summary.Summary":[[8,3,1,"","fit"]],"aggmap.utils.vismap":[[9,4,1,"","plot_grid"],[9,4,1,"","plot_scatter"],[8,4,1,"","plot_tree"]],aggmap:[[9,0,0,"-","AggMapNet"],[7,0,0,"-","aggmodel"],[6,0,0,"-","map"],[9,0,0,"-","show"],[8,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","property","Python property"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:property","3":"py:method","4":"py:function"},terms:{"0":[0,1,3,6,7,8,9,13],"00":[0,1,6,9],"0001":[0,1,6,9],"00fff6":[0,6],"01":[0,1,6,9],"06":8,"1":[0,1,6,7,8,9],"10":[0,1,6,7,8,9],"10000":[0,1,6,8,9],"12":8,"123":[0,6],"128":[0,1,6,8,9],"13":[0,1,6,9],"1300ff":[0,6],"14":[7,8],"15":[0,1,6,9],"16":[0,1,6,7,8,9],"17":[0,1,6,7,8,9],"178b66":[0,6],"18":[0,1,6,9],"1d":[0,1,6,8,9],"1e":[0,1,6,9],"2":[0,1,6,7,8,9],"20":[0,6,8],"200":[0,1,6,9],"2018":8,"2019":[0,6,8],"2020":[0,1,6,9],"2021":[1,7,8,9],"2022":[3,13],"2048":[1,7,9],"21":8,"25":[0,6,8],"255":[0,6],"25ff00":[0,6],"27":8,"29":[0,6,8],"2d":[0,1,3,6,8,9,13],"3":[0,1,3,6,7,8,9,13],"31":8,"32":[0,1,6,9],"36":[0,6,8],"38":7,"4":[0,1,6,8,9],"49":8,"4d":[0,1,6,7,9],"5":[0,1,6,7,9],"50":[3,13],"52":8,"53":[0,1,6,7,9],"54":[7,8],"6":7,"7":[3,7,13],"8":[0,3,6,7,8,13],"8a0075":[0,6],"90":[0,6],"96":[0,6],"class":[0,1,6,7,8,9],"default":[0,1,6,8,9],"do":[1,7,8,9],"float":[0,1,6,9],"function":[0,6,7,8,13],"import":[0,1,3,6,7,8,9,13],"int":[0,1,6,7,8,9],"return":[0,1,6,7,8,9],"true":[0,1,6,7,8,9],A:[0,1,6,9],For:[0,1,6,7,9],If:[0,1,3,6,9,13],The:[0,1,3,6,7,9,13],__:[0,1,6,9],absolut:[1,7,9],acc:[0,1,6,9],access:[1,9],accuraci:[0,1,6,7,9],acid:[3,13],action:7,activ:[0,1,3,6,9,13],actual:8,add_leaf_label:[0,6],after:[0,1,6,9],aggmapnet:[3,13,14],aggmodel:[0,1,6,9,14],aggreg:[3,13],al:[3,13],algorithm:[0,1,6,9],all:[0,1,3,6,9,10,11,13],alppli:7,also:[1,9],an:[0,1,3,6,7,9,13],ani:[0,1,6,7,9],appli:[0,1,6,8,9],apply_logrithm:[0,1,6,7,9],apply_scale_smoth:7,apply_smooth:[0,1,6,7,9],ar:[0,1,3,6,7,9,13],arg:[1,7,8,9],argument:7,arr_1d:[0,1,6,9],arrai:[0,1,6,7,8,9],array_2d:[0,1,6,9],aug:[0,1,6,8,9],author:[0,1,6,7,8,9],auto:[1,7,9],auto_examples_jupyt:[10,11],auto_examples_python:[10,11],automat:[0,1,6,9],averag:[0,1,6,7,9],ax:[0,6],backend:8,backgroud:[1,7,9],base:[0,1,3,6,7,8,9,13],baseestim:[0,6],batch:[0,1,6,9],batch_norm:[0,1,6,9],batch_siz:[0,1,6,9],batch_transform:[0,1,6,9],begin:7,belong:[0,1,6,9],between:8,bidd:[3,13],binari:[0,1,6,9],binary_task:[0,1,6,7,9],block:7,book:[1,7,9],bool:[0,1,6,8,9],bray_curti:8,braycurti:8,by_scipi:[0,1,6,9],c:[1,7,9],calcuat:8,calcul:[0,1,3,6,9,13,14],call:7,callback:7,can:[0,1,3,6,8,9,13],canberra:8,card:[0,1,6,9],categorical_crossentropi:[0,1,6,9],causal:[1,7,9],cbk:[6,14],centroid:[0,1,6,9],chang:7,channel:[0,1,6,7,8,9],channel_col:8,chebyshev:8,chen:[3,13],christophm:[1,7,9],cla_earlystoppingandperform:7,class_num:7,class_weight:[0,6],classes_:[0,1,6,9],classif:7,classifiermixin:[0,6],clean:[0,6],clf:[0,1,6,7,9],clip_text:8,cluster:[0,1,6,9],cluster_channel:[0,1,6,9],cmd:8,cnn:[0,1,3,6,9,13],code:[0,3,6,10,11,13],col:8,color:[0,1,6,9],color_list:[0,6],column:8,com:[3,13],complet:[0,1,6,9],compon:[0,1,6,9],conda:[3,13],conrespond:[0,1,6,9],constant:8,constant_valu:8,contain:[0,1,6,9],content:14,conv1_kernel_s:[0,1,6,9],conv2:8,conv_siz:7,convert:[0,1,6,9],convolut:[0,1,6,9],copi:[0,6],correl:[0,1,3,6,8,9,13],cosin:8,count_trainable_param:7,covert_mpx_to_shapely_df:7,covolut:[0,1,6,9],cpu:8,creat:[0,1,3,6,7,8,9,13],create_print_method:8,criteria:7,cross_entropi:7,current:7,d000ff:[0,6],d:[0,1,6,8,9],dark:[0,6],data:[0,1,3,6,7,8,9,13],datafram:[0,1,6,8,9],deal_list:8,deep:[0,1,6,7,9],defalt:[0,1,6,9],defaut:[1,8,9],dens:[0,1,6,9],dense_avf:[0,1,6,9],dense_lay:[0,1,6,9],descriptors_dist:8,detail:[3,13],develop:[3,13],df:8,dfs_to_tre:8,dfs_to_weightless_newick:8,dfx:[0,1,6,9],dice:8,dict:[0,1,6,7,8,9],disadvantag:[1,7,9],dist_matrix:8,distanc:[6,14],distribut:8,done:8,dot:[1,8,9],download:[10,11],dpi:[0,6],dropout:[0,1,6,9],dure:7,e2ff00:[0,6],e45:[3,13],each:[0,1,6,7,8,9],earli:[0,1,6,9],edu:[0,1,3,6,7,8,9,13],effici:[3,13],element:8,emb_method:[0,1,6,9],embed:[0,1,6,9],embedd:[0,1,6,9],embedding_df:8,emit:8,empti:[0,1,6,9],enabled_data_label:[0,1,6,8,9],end:7,enhanc:[3,13],env:[3,13],epoch:[0,1,6,7,9],error:8,estim:[0,1,6,7,9],et:[3,13],euclidean:8,evalu:[1,7,9],evalul:7,exampl:[7,10,11],expand_nest:[0,6],explain:[1,6,9,14],explain_dev:[6,14],explain_format:[0,1,6,9],explain_model:[0,1,6,9],explan:[1,7,9],extra:[0,1,6,9],extract:8,fail_in_fil:8,fals:[0,1,6,7,8,9],fccde5:[0,6],featur:[0,1,3,6,7,8,9,13],feature_group_list:[0,1,6,9],feb:7,ff0c00:[0,6],ff8800:[0,6],figsiz:[0,6],figur:[1,8,9],file:[0,1,6,8,9],filenam:[0,6],fill:[0,1,6,9],fillnan:[0,1,6,9],fillvalu:8,filter:7,find:[0,1,3,6,9,13],fine:[1,9],fingerprint_dist:8,first:[0,1,6,7,9],fit:[0,1,6,8,9],fmap:[1,3,7,9,13],fmap_shap:[0,1,6,8,9],form:[0,1,6,9],format_exc:8,forward:7,found:[3,13],fri:[1,7,8,9],from:[0,1,6,9],fspecial:8,fspecial_gauss:8,fuction:[0,1,6,8,9],func:8,futur:7,galleri:[10,11],gaussian:[0,1,6,8,9],gen_nwk:[6,14],gener:[3,10,11,13],gennameddist:8,get:[0,1,6,9],get_dat:8,get_datetim:8,get_param:[0,1,6,9],get_shap_valu:7,github:[1,3,7,9,13],given:[0,1,6,9],global:[0,1,3,6,7,9,13],global_explain:[1,7,9],globalimp:7,googl:7,gpl:[3,13],gpu:[0,1,6,9],gpuid:[0,1,6,9],grid:[3,13],group:[0,1,6,8,9],group_color_dict:[0,1,6,9],guid:13,h:[1,7,9],ha:[0,1,6,9],ham:8,handler:8,have:[0,1,3,6,9,13],high:[0,1,6,9],html:[0,1,6,7,8,9],htmlname:[0,1,6,8,9],htmlpath:[0,1,6,8,9],http:[0,1,6,7,9],ident:[1,7,9],identifi:[1,7,9],idx:[1,7,9],imag:[3,13],imapunord:8,impact:[1,7,9],implement:8,imshow:[0,6],imshow_wrap:[0,6],incept:[0,1,6,7,9],includ:[0,1,3,6,8,9,13],increas:[0,1,6,9],index:[3,7,13],infin:8,info_dist:[0,1,6,9],infom:[0,1,6,9],input:[0,1,6,7,8,9],input_data:7,instanc:[0,1,6,9],instead:[1,7,8,9],integ:7,intend:8,intern:[1,9],interpret:[1,7,9],intrins:[3,13],io:[1,7,9],islic:7,isomap:[0,1,6,9],issu:[3,13],iter:8,itol:[0,1,6,9],its:[3,13],j:[3,13],jaccard:8,jupyt:[10,11],k_means_sampl:[1,7,9],kei:[0,1,6,7,9],kera:7,kernel:[0,1,6,9],kernel_s:[0,1,6,7,8,9],kernelexplain:[1,7,9],know:[3,13],kulsinski:8,kwarg:[0,1,6,8,9],l1:8,l2:8,l:8,label:[0,1,6,7,9],larger:[0,1,6,9],last:7,last_avf:[0,1,6,7,9],latter:[0,1,6,9],layer:[0,1,6,9],lead:[1,7,9],leaf_font_s:[0,6],leaf_nam:[0,1,6,8,9],leaf_rot:[0,6],learn:[0,1,6,9],let:[3,13],level:8,like:[0,1,6,9],limiat:[1,7,9],link:[1,7,9],linkag:[0,1,6,9],list:[0,1,3,6,8,9,13],liu:[3,13],lle:[0,1,6,9],lnk_method:[0,1,6,9],load:[0,1,6,9],load_model:[0,1,6,9],local:[0,1,3,6,9,13],local_explain:[1,7,9],localimp:7,locat:[3,13],log:[0,1,6,7,8,9],log_to_fil:8,logarithm:[0,1,6,9],logtool:[6,14],loss:[0,1,6,9,14],low:[0,1,3,6,9,13],lower:[1,7,9],lr:[0,1,6,9],lst:[7,8],m:[0,1,6,9],magnitud:[1,7,9],mai:7,mail:[3,13],main:[0,6],major:[1,3,9,13],male:7,manhattan:8,manhatten:8,mani:[1,7,9],map:[1,3,7,9,13,14],mask:7,math:8,mathemat:[3,13],matlab:8,matric:[0,1,6,9],matrix:8,matrixopt:[6,14],max_i:8,max_work:8,md:[0,1,6,9],mean:[0,1,6,7,9],measur:[1,7,9],memmap:8,method:[0,1,6,7,8,9],metric:[0,1,6,7,9],mimic:8,min:[1,7,9],min_dist:[0,1,6,9],minmax:[0,1,6,9],minmaxscaleclip:[0,6],ml:[1,7,9],mode:[0,6,7,8],model:[0,1,3,6,7,9,13],model_evalu:[0,1,6,9],model_path:[0,1,6,9],modul:[1,3,9,13,14],molmap:[0,1,6,9],monitor:[0,1,6,9],more:[1,7,9],mp2newick:8,mp:[0,1,6,7,8,9],mse:[0,1,6,9],multi:[1,7,8,9],multi_class:[0,1,6,9],multiclass:[0,1,6,9],multiclassestim:[0,1,6,9],multiexecutorrun:8,multilabel:[0,1,6,9],multilabelestim:[0,1,6,9],multinomi:[0,1,6,9],multiproc:[6,14],multiprocessrun:8,multiprocessunorderedbarrun:8,must:[0,1,6,9],mytre:[0,1,6,9],n:[0,1,3,6,7,9,13],n_class:[0,1,6,9],n_compon:[0,6],n_cpu:[0,1,6,8,9],n_featur:[0,1,6,9],n_features_c:[0,1,6,9],n_features_h:[0,1,6,9],n_features_w:[0,1,6,9],n_incept:[0,1,6,9],n_job:[0,1,6,8,9],n_neighbor:[0,1,6,9],n_sampl:[0,1,6,9],naiv:7,name:[0,1,6,7,8,9],nan:[0,1,6,9],nar:[3,13],neg:[0,1,6,9],nest:[0,1,6,9],net:[6,14],network:[3,13],newick:[0,1,6,9],nn:[0,1,6,9],none:[0,1,6,7,8,9],normal:[0,1,6,9],note:[1,7,8,9],notebook:[10,11],notimplementederror:8,nov:8,novel:[3,13],np:8,npydata:8,nsampl:[1,7,9],nu:[0,1,3,6,7,8,9,13],nucleic:[3,13],number:[0,1,6,7,8,9],numpi:[0,1,6,8,9],object:[0,1,6,7,8,9],odd:[0,1,6,8,9],on_epoch_end:7,on_train_begin:7,on_train_end:7,one:[0,1,6,7,8,9],onli:[0,1,6,7,9],oper:8,option:[0,1,6,9],order:[0,1,6,8,9],org:[0,1,6,9],orign:[0,1,6,9],other:[1,7,9],our:[3,13],output:[0,1,3,6,7,8,9,13],overrid:7,packag:[3,13,14],pad:8,page:[3,13],pairwise_dist:8,paper:[3,13],parallel:[0,1,6,9],param:[0,1,6,9],paramet:[0,1,6,7,9],pass:[7,8],path:[1,8,9],patienc:[0,1,6,7,9],pbar:8,pbarhandl:8,perform:[0,1,6,7,9],phenotype_tre:8,pip:[3,13],pipelin:[0,1,6,9],pleas:[0,1,3,6,9,13],plot_grid:[0,1,6,8,9],plot_model:[0,6],plot_scatt:[0,1,6,8,9],plot_tre:[0,6,8],png:[0,6],point:[0,1,6,9],ponit:[0,1,6,9],pos_weight:7,posit:[0,1,6,9],possibl:[0,1,6,9],pprint_tre:8,prc:[0,1,6,9],prc_auc_scor:7,precomput:[0,1,6,9],predict:[0,1,6,7,9],predict_proba:[0,1,6,9],prefix:[1,7,8,9],presum:[3,13],pretti:[0,1,6,9],print:[0,1,6,9],print_debug:8,print_error:8,print_exc:8,print_exc_:8,print_info:8,print_timedelta:8,print_traceback:8,print_warn:8,probabl:[0,1,6,9],problem:[0,1,6,7,9],process:8,processor:8,project:[3,13],prop:7,properti:[0,6],pypi:[3,13],python:[3,10,11,13],r2:[0,1,6,7,9],r2_score:7,radiu:[0,1,6,8,9],rais:8,random:[0,1,6,8,9],random_2dembed:[0,6],random_sampl:8,random_st:[0,1,6,9],rankdir:[0,6],rate:[0,1,6,9],re:[0,1,6,7,9],record:8,refer:[0,1,6,9],refit_c:[0,1,6,8,9],reg_earlystoppingandperform:7,regress:[0,1,6,9],regressionestim:[0,1,6,9],regressormixin:[0,6],relu:[0,1,6,9],remov:[0,1,6,9],requir:[3,13],research:[3,13],reset_handl:8,resnet_block:7,result:[7,8],rmse:[0,1,6,7,9],roc:[0,1,6,7,9],roc_auc:7,rogers_tanimoto:8,rogerstanimoto:8,root:8,run:[7,8],runcmd:8,s:[0,1,6,7,8,9],salienc:[3,13],same:8,sampl:[0,1,3,6,7,8,9,13],sample_weight:[0,1,6,9],sat:8,save:[0,6],save_model:[0,6],scale:[0,1,6,9],scale_method:[0,1,6,9],scatter2arrai:8,scatter2grid:8,scatter:[0,1,6,8,9],scikit:[0,1,6,9],score:[0,1,3,6,9,13],se:[0,1,6,9],search:[3,13],select:[0,1,6,9],self:[0,1,6,9],sep:[1,7,8,9],separ:[1,9],set:[0,1,6,7,9],set_level:8,set_param:[0,1,6,9],set_text_length:8,sever:[0,1,6,9],shap:[1,7,9],shape:[0,1,6,7,8,9],shape_valu:[1,7,9],shaplei:[1,3,7,9,13],shapley_explain:[1,7,9],shen:[0,1,3,6,7,8,9,13],shenwanxiang:[3,8,13],should:[7,8],show:14,show_layer_nam:[0,6],show_shap:[0,6],sigma:[0,1,6,7,8,9],sigmoid:7,sigmoid_cross_entropy_with_logit:[0,1,6,9],sigmoid_cross_entropy_with_logits_v2:[0,1,6,9],sigmoidi:7,simpl:[0,1,3,6,9,13],simpli:[3,13],simply_explain:7,sinc:[0,1,6,9],singl:[0,1,6,9],size:[0,1,6,8,9],sklearn:[0,6],smartpad:8,smooth:[0,1,6,9],smoth:7,so:[0,1,6,8,9],softmax:[0,1,6,9],softwar:[3,13],sokal_sneath:8,some:[0,1,6,9],sourc:[0,1,3,6,7,8,9,10,11,13],spatial:[3,13],specif:[0,1,6,9],specifi:8,sphinx:[10,11],split:[0,1,6,8,9],split_channel:[0,1,6,8,9],sqeuclidean:8,sqrt:8,stabl:[0,1,6,9],standard:[0,1,6,8,9],standardscal:[0,6],statist:8,statu:8,stderr:8,stdout:8,stop:[0,1,6,9],str:[0,1,6,8,9],stride:7,string:[0,1,6,9],structur:[3,13],subclass:[7,8],submit:[0,1,6,9],submodul:14,subobject:[0,1,6,9],subpackag:14,success:8,suffix:8,sum_i:8,summari:[6,14],summary2:8,sun:[0,1,6,8,9],supervis:[3,13],t:[0,1,6,8,9],take:8,target:8,target_s:8,task:[0,1,6,9],task_typ:7,taxicab:8,tb:[0,6],test:[0,1,6,9],testi:7,testx:7,text:8,tf:[0,1,6,9],than:[0,1,6,9],thei:[0,1,3,6,9,13],them:[0,1,6,9],theree:[3,13],thi:[0,1,6,7,8,9],thread:8,threshold:[0,1,6,9],time:[1,7,9],to_fil:[0,6],to_nwk_tre:[0,1,6,9],tqdm:8,tqdm_arg:8,tracker:[3,13],train:[0,1,6,7,9],train_data:7,transform:[0,1,6,8,9],transform_mpx_to_df:[0,1,6,9],tree:[0,1,6,8,9],tree_add:8,tree_inst:8,tree_to_newick:8,treefil:[0,1,6,8,9],trian:[0,1,6,9],tsne:[0,1,6,9],tue:[0,1,6,7,9],tune:[1,9],tupl:[0,1,6,8,9],two:[1,9],type:[0,1,6,9],u:[0,1,3,6,7,8,9,13],umap:[0,1,6,9],under:[3,13],underli:[3,13],unit:[7,8],unord:[3,13],unstructur:[3,13],unsupervis:[3,13],updat:[0,1,6,9],upgrad:[3,13],us:[0,3,6,7,8,13],usecas:8,util:[1,6,9,14],val_:7,val_loss:[0,1,6,7,9],val_metr:[0,1,6,9],val_r2:[0,1,6,9],valid:7,valid_data:7,valu:[0,1,6,7,8,9],var_thr:[0,1,6,9],vari:8,varianc:[0,1,6,7,9],variou:[0,1,6,9],vector:[0,1,6,8,9],vector_1d:8,verbos:[0,1,6,7,8,9],version:8,vismap:[1,6,9,14],vmax:[0,6],vmin:[0,6],w:[1,3,7,9,13],wanxiang:[0,1,3,6,7,8,9,13],we:[0,1,3,6,8,9,13],wed:8,weight:[0,1,6,9],weighted_cross_entropi:7,well:[0,1,6,9],whatev:8,when:[1,7,9],where:[0,1,6,7,9],whether:[0,1,6,9],which:[3,13],wish:8,work:[0,1,6,9],wrt:[0,1,6,9],x:[0,1,3,6,7,8,9,13],x_:[1,7,9],x_arr:[0,6],x_i:8,x_max:[0,6],x_valid:[0,6],xmax:[0,6],xmean:[0,6],xmin:[0,6],xstd:[0,6],y:[0,1,3,6,7,8,9,13],y_:7,y_i:8,y_ob:7,y_pred:7,y_score:7,y_true:7,y_valid:[0,6],yet:[3,13],you:[3,8,13],zip:[10,11]},titles:["aggmap package","aggmap API Guide","&lt;no title&gt;","Jigsaw-like aggmap: A Robust and Explainable Multi-Channel Omics Deep Learning Tool","aggmap","&lt;no title&gt;","aggmap package","aggmap.aggmodel package","aggmap.utils package","aggmap API Guide","&lt;no title&gt;","&lt;no title&gt;","&lt;no title&gt;","Jigsaw-like aggmap: A Robust and Explainable Multi-Channel Omics Deep Learning Tool","aggmap","&lt;no title&gt;"],titleterms:{"function":[1,9],A:[3,13],aggmap:[0,1,3,4,6,7,8,9,13,14],aggmapnet:[0,1,6,9],aggmodel:7,api:[1,9,13],calcul:8,cbk:7,channel:[3,13],content:[0,3,6,7,8,13],contribut:[3,13],deep:[3,13],distanc:8,explain:[3,7,13],explain_dev:7,gen_nwk:8,guid:[1,9],indic:[3,13],instal:[3,13],jigsaw:[3,13],learn:[3,13],licens:[3,13],like:[3,13],logtool:8,loss:7,map:[0,6],matrixopt:8,modul:[0,6,7,8],multi:[3,13],multiproc:8,net:7,omic:[3,13],packag:[0,6,7,8],refer:13,robust:[3,13],show:[0,6],submodul:[0,6,7,8],subpackag:[0,6],summari:8,support:[3,13],tabl:[3,13],tool:[3,13],us:[1,9],util:8,vismap:8}})