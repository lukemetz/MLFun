Function profiling
==================
  Message: /usr/local/lib/python2.7/dist-packages/blocks-0.0.1-py2.7.egg/blocks/monitoring/evaluators.py:166
  Time in 2 calls to Function.__call__: 2.102852e-03s
  Time in Function.fn.__call__: 2.052307e-03s (97.596%)
  Time in thunks: 1.898050e-03s (90.261%)
  Total compile time: 5.679297e-02s
    Number of Apply nodes: 6
    Theano Optimizer time: 1.254606e-02s
       Theano validate time: 2.176762e-04s
    Theano Linker time (includes C, CUDA code generation/compiling): 3.267217e-02s
       Import time 2.614260e-03s

Time in all call to theano.grad() 6.174994e-02s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  100.0%   100.0%       0.002s       1.58e-04s     C       12       6   theano.compile.ops.DeepCopyOp
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  100.0%   100.0%       0.002s       1.58e-04s     C       12        6   DeepCopyOp
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  92.3%    92.3%       0.002s       8.76e-04s      2     3   DeepCopyOp(CudaNdarrayConstant{0.0})
   2.6%    94.8%       0.000s       2.46e-05s      2     2   DeepCopyOp(CudaNdarrayConstant{0.0})
   2.2%    97.0%       0.000s       2.09e-05s      2     1   DeepCopyOp(CudaNdarrayConstant{0.0})
   2.1%    99.1%       0.000s       1.99e-05s      2     0   DeepCopyOp(CudaNdarrayConstant{0.0})
   0.6%    99.7%       0.000s       5.60e-06s      2     5   DeepCopyOp(TensorConstant{0.0})
   0.3%   100.0%       0.000s       2.50e-06s      2     4   DeepCopyOp(TensorConstant{0.0})
   ... (remaining 0 Apply instances account for 0.00%(0.00s) of the runtime)

Optimizer Profile
-----------------
 SeqOptimizer  time 0.012s for 8/0 nodes before/after optimization
   0.000s for fgraph.validate()
   0.002s for callback
   time      - (name, class, index) - validate time
   0.003589s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.000s
     EquilibriumOptimizer      canonicalize
       time 0.003s for 3 passes
       nb nodes (start, end,  max) 8 0 8
       time io_toposort 0.000s
       time in local optimizers 0.003s
       time in global optimizers 0.000s
        0 - 0.001s 4 (0.000s in global opts, 0.000s io_toposort) - 8 nodes - ('local_fill_to_alloc', 4)
        1 - 0.002s 2 (0.000s in global opts, 0.000s io_toposort) - 1 nodes - ('MergeOptimizer', 1) ('constant_folding', 1)
        2 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       times - times applied - nb node created - name:
       0.002s - 1 - 0 - constant_folding
       0.001s - 4 - 0 - local_fill_to_alloc
       0.000s - 1 - 0 - MergeOptimizer
       0.000s - in 74 optimization that where not used (display only those with a runtime > 0)
         0.000s - local_cut_gpu_host_gpu
         0.000s - local_upcast_elemwise_constant_inputs
         0.000s - local_func_inv
         0.000s - local_useless_elemwise
         0.000s - local_fill_sink
         0.000s - local_track_shape_i
         0.000s - local_fill_cut
         0.000s - local_cast_cast
         0.000s - local_remove_switch_const_cond

   0.002741s - ('add_destroy_handler', 'AddDestroyHandler', 18) - 0.000s
   0.000754s - ('scan_eqopt1', 'EquilibriumOptimizer', 2) - 0.000s
     EquilibriumOptimizer      scan_eqopt1
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 8 8 8
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.001s
        0 - 0.001s 0 (0.001s in global opts, 0.000s io_toposort) - 8 nodes - 
   0.000694s - ('merge3', 'MergeOptimizer', 40) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 0.000688076019287
       validate_time 0.000135898590088
       callback_time 0.000181913375854
       nb_merged 4
       nb_constant 4
   0.000487s - ('gpu_opt', 'SeqOptimizer', 12) - 0.000s
     SeqOptimizer      gpu_opt  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000355s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000064s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000039s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000019s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000448s - ('gpu_after_fusion', 'SeqOptimizer', 17) - 0.000s
     SeqOptimizer      gpu_after_fusion  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000345s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000066s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000024s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000003s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000337s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
     SeqOptimizer      BlasOpt  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000100s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
         GemmOptimizer
          nb_iter 1
          nb_replacement 0
          nb_replacement_didn_t_remove 0
          nb_inconsistency_make 0
          nb_inconsistency_replace 0
          time_canonicalize 0
          time_factor_can 0
          time_factor_list 0
          time_toposort 1.59740447998e-05
          validate_time 0.0
          callback_time 0.0
       0.000075s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          local_gemm_to_gemv
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000040s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 1.50203704834e-05
           loop time 1.19209289551e-06
           callback_time 0.0
       0.000039s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 1.59740447998e-05
           loop time 9.53674316406e-07
           callback_time 0.0
       0.000035s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 1.50203704834e-05
           loop time 9.53674316406e-07
           callback_time 0.0
       0.000034s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 1.4066696167e-05
           loop time 1.19209289551e-06
           callback_time 0.0

   0.000322s - ('merge1', 'MergeOptimizer', 0) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 1.00135803223e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000298s - ('ShapeOpt', 'ShapeOptimizer', 1) - 0.000s
   0.000286s - ('scan_eqopt2', 'EquilibriumOptimizer', 7) - 0.000s
     EquilibriumOptimizer      scan_eqopt2
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000229s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
     EquilibriumOptimizer      specialize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000223s - ('elemwise_fusion', 'SeqOptimizer', 14) - 0.000s
     SeqOptimizer      elemwise_fusion  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000113s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 2.86102294922e-06
       0.000103s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 1.90734863281e-06

   0.000181s - ('stabilize', 'EquilibriumOptimizer', 6) - 0.000s
     EquilibriumOptimizer      stabilize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000157s - ('cond_make_inplace', 'TopoOptimizer', 36) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 0.000132083892822
       loop time 0.0
       callback_time 0.0
   0.000145s - ('gpua_elemwise_fusion', 'FusionOptimizer', 29) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 3.09944152832e-06
   0.000128s - ('local_IncSubtensor_serialize', 'TopoOptimizer', 3) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (8, 8, 0)
       init io_toposort 7.39097595215e-05
       loop time 3.00407409668e-05
       callback_time 0.0
   0.000127s - ('blas_opt_inplace', 'TopoOptimizer', 22) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.78813934326e-05
       loop time 0.0
       callback_time 0.0
   0.000102s - ('gpuablas_opt_inplace', 'TopoOptimizer', 24) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.59740447998e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000102s - ('InplaceGpuBlasOpt', 'TopoOptimizer', 23) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.71661376953e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000099s - ('gpu_elemwise_fusion', 'FusionOptimizer', 15) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 9.53674316406e-07
   0.000094s - ('merge2', 'MergeOptimizer', 16) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 9.01222229004e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000092s - ('local_dnn_conv_inplace', 'TopoOptimizer', 26) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.62124633789e-05
       loop time 0.0
       callback_time 0.0
   0.000077s - ('uncanonicalize', 'EquilibriumOptimizer', 11) - 0.000s
     EquilibriumOptimizer      uncanonicalize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000068s - ('local_gemm16_inplace', 'TopoOptimizer', 27) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.81198120117e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000065s - ('local_inplace_incsubtensor1', 'TopoOptimizer', 20) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.78813934326e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000062s - ('specialize_device', 'EquilibriumOptimizer', 13) - 0.000s
     EquilibriumOptimizer      specialize_device
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000060s - ('local_inplace_setsubtensor', 'TopoOptimizer', 21) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.69277191162e-05
       loop time 1.19209289551e-06
       callback_time 0.0
   0.000055s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.000s
   0.000048s - ('dimshuffle_as_view', 'TopoOptimizer', 19) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 2.00271606445e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000045s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 32) - 0.000s
   0.000042s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 34) - 0.000s
   0.000040s - ('random_make_inplace', 'TopoOptimizer', 38) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.59740447998e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000038s - ('local_destructive', 'TopoOptimizer', 37) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.59740447998e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000038s - ('make_ger_destructive', 'TopoOptimizer', 28) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 0.0
       callback_time 0.0
   0.000037s - ('c_blas_destructive', 'TopoOptimizer', 25) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000035s - ('mrg_random_make_inplace', 'TopoOptimizer', 39) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.4066696167e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000032s - ('gpu_scanOp_make_inplace', 'ScanInplaceOptimizer', 31) - 0.000s
   0.000031s - ('merge1.2', 'MergeOptimizer', 5) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 2.71797180176e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000028s - ('gpua_scanOp_make_inplace', 'ScanInplaceOptimizer', 33) - 0.000s
   0.000027s - ('scanOp_make_inplace', 'ScanInplaceOptimizer', 35) - 0.000s
   0.000007s - ('crossentropy_to_crossentropy_with_softmax', 'FromFunctionOptimizer', 10) - 0.000s

Function profiling
==================
  Message: /usr/local/lib/python2.7/dist-packages/blocks-0.0.1-py2.7.egg/blocks/monitoring/evaluators.py:176
  Time in 1 calls to Function.__call__: 5.912781e-05s
  Time in Function.fn.__call__: 4.315376e-05s (72.984%)
  Time in thunks: 3.075600e-05s (52.016%)
  Total compile time: 5.363870e-01s
    Number of Apply nodes: 4
    Theano Optimizer time: 9.371042e-03s
       Theano validate time: 0.000000e+00s
    Theano Linker time (includes C, CUDA code generation/compiling): 5.192809e-01s
       Import time 5.919933e-04s

Time in all call to theano.grad() 6.174994e-02s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  90.7%    90.7%       0.000s       1.39e-05s     C        2       2   theano.sandbox.cuda.basic_ops.HostFromGpu
   9.3%   100.0%       0.000s       1.43e-06s     C        2       2   theano.tensor.elemwise.Elemwise
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  90.7%    90.7%       0.000s       1.39e-05s     C        2        2   HostFromGpu
   9.3%   100.0%       0.000s       1.43e-06s     C        2        2   Elemwise{true_div,no_inplace}
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  58.1%    58.1%       0.000s       1.79e-05s      1     1   HostFromGpu(shared_None)
  32.6%    90.7%       0.000s       1.00e-05s      1     0   HostFromGpu(shared_None)
   6.2%    96.9%       0.000s       1.91e-06s      1     3   Elemwise{true_div,no_inplace}(shared_cost, HostFromGpu.0)
   3.1%   100.0%       0.000s       9.54e-07s      1     2   Elemwise{true_div,no_inplace}(shared_misclassification, HostFromGpu.0)
   ... (remaining 0 Apply instances account for 0.00%(0.00s) of the runtime)

Optimizer Profile
-----------------
 SeqOptimizer  time 0.009s for 4/4 nodes before/after optimization
   0.000s for fgraph.validate()
   0.000s for callback
   time      - (name, class, index) - validate time
   0.000949s - ('add_destroy_handler', 'AddDestroyHandler', 18) - 0.000s
   0.000735s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.000s
     EquilibriumOptimizer      canonicalize
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.001s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000618s - ('gpu_opt', 'SeqOptimizer', 12) - 0.000s
     SeqOptimizer      gpu_opt  time 0.001s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000417s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000143s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000042s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000008s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000586s - ('scan_eqopt2', 'EquilibriumOptimizer', 7) - 0.000s
     EquilibriumOptimizer      scan_eqopt2
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.001s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000583s - ('gpu_after_fusion', 'SeqOptimizer', 17) - 0.000s
     SeqOptimizer      gpu_after_fusion  time 0.001s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000400s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000142s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000030s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000002s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000537s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
     SeqOptimizer      BlasOpt  time 0.001s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000126s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
         GemmOptimizer
          nb_iter 1
          nb_replacement 0
          nb_replacement_didn_t_remove 0
          nb_inconsistency_make 0
          nb_inconsistency_replace 0
          time_canonicalize 0
          time_factor_can 0
          time_factor_list 0
          time_toposort 4.00543212891e-05
          validate_time 0.0
          callback_time 0.0
       0.000101s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          local_gemm_to_gemv
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000084s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 4.10079956055e-05
           loop time 1.90734863281e-05
           callback_time 0.0
       0.000074s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.91006469727e-05
           loop time 1.28746032715e-05
           callback_time 0.0
       0.000069s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.79085540771e-05
           loop time 9.05990600586e-06
           callback_time 0.0
       0.000067s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.88622283936e-05
           loop time 8.10623168945e-06
           callback_time 0.0

   0.000505s - ('scan_eqopt1', 'EquilibriumOptimizer', 2) - 0.000s
     EquilibriumOptimizer      scan_eqopt1
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000495s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
     EquilibriumOptimizer      specialize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000397s - ('elemwise_fusion', 'SeqOptimizer', 14) - 0.000s
     SeqOptimizer      elemwise_fusion  time 0.000s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000214s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 6.29425048828e-05
       0.000175s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 6.60419464111e-05

   0.000371s - ('stabilize', 'EquilibriumOptimizer', 6) - 0.000s
     EquilibriumOptimizer      stabilize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000242s - ('gpua_elemwise_fusion', 'FusionOptimizer', 29) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 7.39097595215e-05
   0.000219s - ('ShapeOpt', 'ShapeOptimizer', 1) - 0.000s
   0.000174s - ('cond_make_inplace', 'TopoOptimizer', 36) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 0.000138998031616
       loop time 9.05990600586e-06
       callback_time 0.0
   0.000172s - ('gpu_elemwise_fusion', 'FusionOptimizer', 15) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 6.79492950439e-05
   0.000144s - ('gpuablas_opt_inplace', 'TopoOptimizer', 24) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.29153442383e-05
       loop time 2.21729278564e-05
       callback_time 0.0
   0.000142s - ('InplaceGpuBlasOpt', 'TopoOptimizer', 23) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 3.981590271e-05
       loop time 2.40802764893e-05
       callback_time 0.0
   0.000142s - ('gpu_scanOp_make_inplace', 'ScanInplaceOptimizer', 31) - 0.000s
   0.000139s - ('merge3', 'MergeOptimizer', 40) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 0.000133991241455
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000139s - ('blas_opt_inplace', 'TopoOptimizer', 22) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.69277191162e-05
       callback_time 0.0
   0.000134s - ('local_dnn_conv_inplace', 'TopoOptimizer', 26) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.31537628174e-05
       loop time 1.19209289551e-05
       callback_time 0.0
   0.000133s - ('merge1', 'MergeOptimizer', 0) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 9.05990600586e-06
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000129s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 34) - 0.000s
   0.000109s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.000s
   0.000105s - ('uncanonicalize', 'EquilibriumOptimizer', 11) - 0.000s
     EquilibriumOptimizer      uncanonicalize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000098s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 32) - 0.000s
   0.000097s - ('gpua_scanOp_make_inplace', 'ScanInplaceOptimizer', 33) - 0.000s
   0.000097s - ('local_inplace_incsubtensor1', 'TopoOptimizer', 20) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.48226928711e-05
       loop time 5.00679016113e-06
       callback_time 0.0
   0.000096s - ('specialize_device', 'EquilibriumOptimizer', 13) - 0.000s
     EquilibriumOptimizer      specialize_device
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000095s - ('local_gemm16_inplace', 'TopoOptimizer', 27) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.09672546387e-05
       callback_time 0.0
   0.000094s - ('scanOp_make_inplace', 'ScanInplaceOptimizer', 35) - 0.000s
   0.000094s - ('local_inplace_setsubtensor', 'TopoOptimizer', 21) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 8.10623168945e-06
       callback_time 0.0
   0.000091s - ('merge2', 'MergeOptimizer', 16) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 8.70227813721e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000081s - ('dimshuffle_as_view', 'TopoOptimizer', 19) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.79221343994e-05
       loop time 8.10623168945e-06
       callback_time 0.0
   0.000080s - ('local_IncSubtensor_serialize', 'TopoOptimizer', 3) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.69277191162e-05
       callback_time 0.0
   0.000079s - ('mrg_random_make_inplace', 'TopoOptimizer', 39) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 3.91006469727e-05
       loop time 1.69277191162e-05
       callback_time 0.0
   0.000076s - ('local_destructive', 'TopoOptimizer', 37) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.19209289551e-05
       callback_time 0.0
   0.000075s - ('random_make_inplace', 'TopoOptimizer', 38) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 1.09672546387e-05
       callback_time 0.0
   0.000075s - ('c_blas_destructive', 'TopoOptimizer', 25) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 3.981590271e-05
       loop time 1.31130218506e-05
       callback_time 0.0
   0.000073s - ('crossentropy_to_crossentropy_with_softmax', 'FromFunctionOptimizer', 10) - 0.000s
   0.000069s - ('make_ger_destructive', 'TopoOptimizer', 28) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 3.88622283936e-05
       loop time 8.10623168945e-06
       callback_time 0.0
   0.000027s - ('merge1.2', 'MergeOptimizer', 5) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 2.19345092773e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0

Function profiling
==================
  Message: /usr/local/lib/python2.7/dist-packages/blocks-0.0.1-py2.7.egg/blocks/monitoring/evaluators.py:166
  Time in 2 calls to Function.__call__: 2.948284e-03s
  Time in Function.fn.__call__: 2.901793e-03s (98.423%)
  Time in thunks: 2.727032e-03s (92.496%)
  Total compile time: 3.063798e-02s
    Number of Apply nodes: 6
    Theano Optimizer time: 1.304817e-02s
       Theano validate time: 2.534389e-04s
    Theano Linker time (includes C, CUDA code generation/compiling): 5.553007e-03s
       Import time 0.000000e+00s

Time in all call to theano.grad() 6.174994e-02s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  100.0%   100.0%       0.003s       2.27e-04s     C       12       6   theano.compile.ops.DeepCopyOp
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  100.0%   100.0%       0.003s       2.27e-04s     C       12        6   DeepCopyOp
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  94.7%    94.7%       0.003s       1.29e-03s      2     3   DeepCopyOp(CudaNdarrayConstant{0.0})
   1.6%    96.3%       0.000s       2.21e-05s      2     1   DeepCopyOp(CudaNdarrayConstant{0.0})
   1.6%    97.9%       0.000s       2.19e-05s      2     2   DeepCopyOp(CudaNdarrayConstant{0.0})
   1.5%    99.4%       0.000s       2.00e-05s      2     0   DeepCopyOp(CudaNdarrayConstant{0.0})
   0.4%    99.8%       0.000s       5.48e-06s      2     5   DeepCopyOp(TensorConstant{0.0})
   0.2%   100.0%       0.000s       2.50e-06s      2     4   DeepCopyOp(TensorConstant{0.0})
   ... (remaining 0 Apply instances account for 0.00%(0.00s) of the runtime)

Optimizer Profile
-----------------
 SeqOptimizer  time 0.013s for 8/0 nodes before/after optimization
   0.000s for fgraph.validate()
   0.003s for callback
   time      - (name, class, index) - validate time
   0.003295s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.000s
     EquilibriumOptimizer      canonicalize
       time 0.003s for 3 passes
       nb nodes (start, end,  max) 8 0 8
       time io_toposort 0.000s
       time in local optimizers 0.003s
       time in global optimizers 0.000s
        0 - 0.001s 4 (0.000s in global opts, 0.000s io_toposort) - 8 nodes - ('local_fill_to_alloc', 4)
        1 - 0.002s 2 (0.000s in global opts, 0.000s io_toposort) - 1 nodes - ('MergeOptimizer', 1) ('constant_folding', 1)
        2 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       times - times applied - nb node created - name:
       0.002s - 1 - 0 - constant_folding
       0.001s - 4 - 0 - local_fill_to_alloc
       0.000s - 1 - 0 - MergeOptimizer
       0.000s - in 74 optimization that where not used (display only those with a runtime > 0)
         0.000s - local_cut_gpu_host_gpu
         0.000s - local_upcast_elemwise_constant_inputs
         0.000s - local_useless_elemwise
         0.000s - local_func_inv
         0.000s - local_fill_sink
         0.000s - local_track_shape_i
         0.000s - local_cast_cast
         0.000s - local_remove_switch_const_cond
         0.000s - local_fill_cut

   0.002881s - ('add_destroy_handler', 'AddDestroyHandler', 18) - 0.000s
   0.000996s - ('merge3', 'MergeOptimizer', 40) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 0.00098705291748
       validate_time 0.000143766403198
       callback_time 0.000373840332031
       nb_merged 4
       nb_constant 4
   0.000759s - ('scan_eqopt1', 'EquilibriumOptimizer', 2) - 0.000s
     EquilibriumOptimizer      scan_eqopt1
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 8 8 8
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.001s
        0 - 0.001s 0 (0.001s in global opts, 0.000s io_toposort) - 8 nodes - 
   0.000497s - ('gpu_opt', 'SeqOptimizer', 12) - 0.000s
     SeqOptimizer      gpu_opt  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000349s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000065s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000060s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000010s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000481s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
     SeqOptimizer      BlasOpt  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000123s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          local_gemm_to_gemv
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000117s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
         GemmOptimizer
          nb_iter 1
          nb_replacement 0
          nb_replacement_didn_t_remove 0
          nb_inconsistency_make 0
          nb_inconsistency_replace 0
          time_canonicalize 0
          time_factor_can 0
          time_factor_list 0
          time_toposort 1.59740447998e-05
          validate_time 0.0
          callback_time 0.0
       0.000066s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 2.59876251221e-05
           loop time 9.53674316406e-07
           callback_time 0.0
       0.000059s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 2.28881835938e-05
           loop time 9.53674316406e-07
           callback_time 0.0
       0.000058s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 2.31266021729e-05
           loop time 9.53674316406e-07
           callback_time 0.0
       0.000037s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (0, 0, 0)
           init io_toposort 1.50203704834e-05
           loop time 9.53674316406e-07
           callback_time 0.0

   0.000475s - ('gpu_after_fusion', 'SeqOptimizer', 17) - 0.000s
     SeqOptimizer      gpu_after_fusion  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000356s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000073s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 0 0 0
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
       0.000031s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000004s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000371s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
     EquilibriumOptimizer      specialize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000335s - ('merge1', 'MergeOptimizer', 0) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 1.09672546387e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000296s - ('ShapeOpt', 'ShapeOptimizer', 1) - 0.000s
   0.000290s - ('scan_eqopt2', 'EquilibriumOptimizer', 7) - 0.000s
     EquilibriumOptimizer      scan_eqopt2
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000220s - ('elemwise_fusion', 'SeqOptimizer', 14) - 0.000s
     SeqOptimizer      elemwise_fusion  time 0.000s for 0/0 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000112s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 3.09944152832e-06
       0.000101s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 1.90734863281e-06

   0.000184s - ('stabilize', 'EquilibriumOptimizer', 6) - 0.000s
     EquilibriumOptimizer      stabilize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000159s - ('gpua_elemwise_fusion', 'FusionOptimizer', 29) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 3.81469726562e-06
   0.000133s - ('uncanonicalize', 'EquilibriumOptimizer', 11) - 0.000s
     EquilibriumOptimizer      uncanonicalize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000129s - ('local_IncSubtensor_serialize', 'TopoOptimizer', 3) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (8, 8, 0)
       init io_toposort 7.70092010498e-05
       loop time 2.90870666504e-05
       callback_time 0.0
   0.000126s - ('blas_opt_inplace', 'TopoOptimizer', 22) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.8835067749e-05
       loop time 0.0
       callback_time 0.0
   0.000102s - ('local_dnn_conv_inplace', 'TopoOptimizer', 26) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.71661376953e-05
       loop time 0.0
       callback_time 0.0
   0.000102s - ('InplaceGpuBlasOpt', 'TopoOptimizer', 23) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.81198120117e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000100s - ('gpu_elemwise_fusion', 'FusionOptimizer', 15) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 9.53674316406e-07
   0.000099s - ('gpuablas_opt_inplace', 'TopoOptimizer', 24) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.69277191162e-05
       loop time 0.0
       callback_time 0.0
   0.000095s - ('merge2', 'MergeOptimizer', 16) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 9.10758972168e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000082s - ('specialize_device', 'EquilibriumOptimizer', 13) - 0.000s
     EquilibriumOptimizer      specialize_device
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 0 0 0
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 0 nodes - 
   0.000079s - ('local_inplace_setsubtensor', 'TopoOptimizer', 21) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.78813934326e-05
       loop time 1.19209289551e-06
       callback_time 0.0
   0.000069s - ('local_inplace_incsubtensor1', 'TopoOptimizer', 20) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.90734863281e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000061s - ('local_gemm16_inplace', 'TopoOptimizer', 27) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.59740447998e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000056s - ('dimshuffle_as_view', 'TopoOptimizer', 19) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 2.09808349609e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000052s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.000s
   0.000043s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 34) - 0.000s
   0.000042s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 32) - 0.000s
   0.000039s - ('cond_make_inplace', 'TopoOptimizer', 36) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000038s - ('c_blas_destructive', 'TopoOptimizer', 25) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000036s - ('random_make_inplace', 'TopoOptimizer', 38) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000036s - ('local_destructive', 'TopoOptimizer', 37) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 0.0
       callback_time 0.0
   0.000036s - ('make_ger_destructive', 'TopoOptimizer', 28) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000035s - ('mrg_random_make_inplace', 'TopoOptimizer', 39) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (0, 0, 0)
       init io_toposort 1.50203704834e-05
       loop time 9.53674316406e-07
       callback_time 0.0
   0.000033s - ('gpu_scanOp_make_inplace', 'ScanInplaceOptimizer', 31) - 0.000s
   0.000032s - ('merge1.2', 'MergeOptimizer', 5) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 2.69412994385e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000028s - ('gpua_scanOp_make_inplace', 'ScanInplaceOptimizer', 33) - 0.000s
   0.000027s - ('scanOp_make_inplace', 'ScanInplaceOptimizer', 35) - 0.000s
   0.000013s - ('crossentropy_to_crossentropy_with_softmax', 'FromFunctionOptimizer', 10) - 0.000s

Function profiling
==================
  Message: /usr/local/lib/python2.7/dist-packages/blocks-0.0.1-py2.7.egg/blocks/monitoring/evaluators.py:176
  Time in 1 calls to Function.__call__: 5.912781e-05s
  Time in Function.fn.__call__: 3.910065e-05s (66.129%)
  Time in thunks: 2.908707e-05s (49.194%)
  Total compile time: 2.002001e-02s
    Number of Apply nodes: 4
    Theano Optimizer time: 9.593010e-03s
       Theano validate time: 0.000000e+00s
    Theano Linker time (includes C, CUDA code generation/compiling): 3.047943e-03s
       Import time 0.000000e+00s

Time in all call to theano.grad() 6.174994e-02s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  89.3%    89.3%       0.000s       1.30e-05s     C        2       2   theano.sandbox.cuda.basic_ops.HostFromGpu
  10.7%   100.0%       0.000s       1.55e-06s     C        2       2   theano.tensor.elemwise.Elemwise
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  89.3%    89.3%       0.000s       1.30e-05s     C        2        2   HostFromGpu
  10.7%   100.0%       0.000s       1.55e-06s     C        2        2   Elemwise{true_div,no_inplace}
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  54.9%    54.9%       0.000s       1.60e-05s      1     1   HostFromGpu(shared_None)
  34.4%    89.3%       0.000s       1.00e-05s      1     0   HostFromGpu(shared_None)
   6.6%    95.9%       0.000s       1.91e-06s      1     3   Elemwise{true_div,no_inplace}(shared_cost, HostFromGpu.0)
   4.1%   100.0%       0.000s       1.19e-06s      1     2   Elemwise{true_div,no_inplace}(shared_misclassification, HostFromGpu.0)
   ... (remaining 0 Apply instances account for 0.00%(0.00s) of the runtime)

Optimizer Profile
-----------------
 SeqOptimizer  time 0.010s for 4/4 nodes before/after optimization
   0.000s for fgraph.validate()
   0.000s for callback
   time      - (name, class, index) - validate time
   0.000971s - ('add_destroy_handler', 'AddDestroyHandler', 18) - 0.000s
   0.000856s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.000s
     EquilibriumOptimizer      canonicalize
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.001s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000717s - ('gpu_opt', 'SeqOptimizer', 12) - 0.000s
     SeqOptimizer      gpu_opt  time 0.001s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000410s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000253s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000040s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000005s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000599s - ('scan_eqopt1', 'EquilibriumOptimizer', 2) - 0.000s
     EquilibriumOptimizer      scan_eqopt1
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.001s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000585s - ('gpu_after_fusion', 'SeqOptimizer', 17) - 0.000s
     SeqOptimizer      gpu_after_fusion  time 0.001s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000396s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000147s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000029s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000003s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.000584s - ('scan_eqopt2', 'EquilibriumOptimizer', 7) - 0.000s
     EquilibriumOptimizer      scan_eqopt2
       time 0.001s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.001s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000539s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
     SeqOptimizer      BlasOpt  time 0.001s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000125s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
         GemmOptimizer
          nb_iter 1
          nb_replacement 0
          nb_replacement_didn_t_remove 0
          nb_inconsistency_make 0
          nb_inconsistency_replace 0
          time_canonicalize 0
          time_factor_can 0
          time_factor_list 0
          time_toposort 3.981590271e-05
          validate_time 0.0
          callback_time 0.0
       0.000105s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          local_gemm_to_gemv
           time 0.000s for 1 passes
           nb nodes (start, end,  max) 4 4 4
           time io_toposort 0.000s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
       0.000087s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.79085540771e-05
           loop time 1.19209289551e-05
           callback_time 0.0
       0.000077s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.79085540771e-05
           loop time 1.59740447998e-05
           callback_time 0.0
       0.000066s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.69548797607e-05
           loop time 9.05990600586e-06
           callback_time 0.0
       0.000065s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (4, 4, 0)
           init io_toposort 3.81469726562e-05
           loop time 5.96046447754e-06
           callback_time 0.0

   0.000505s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
     EquilibriumOptimizer      specialize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000394s - ('elemwise_fusion', 'SeqOptimizer', 14) - 0.000s
     SeqOptimizer      elemwise_fusion  time 0.000s for 4/4 nodes before/after optimization
       0.000s for fgraph.validate()
       0.000s for callback
       0.000214s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 6.22272491455e-05
       0.000173s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 6.60419464111e-05

   0.000361s - ('stabilize', 'EquilibriumOptimizer', 6) - 0.000s
     EquilibriumOptimizer      stabilize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000319s - ('ShapeOpt', 'ShapeOptimizer', 1) - 0.000s
   0.000215s - ('gpua_elemwise_fusion', 'FusionOptimizer', 29) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 7.29560852051e-05
   0.000196s - ('merge1', 'MergeOptimizer', 0) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 1.38282775879e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000165s - ('gpu_elemwise_fusion', 'FusionOptimizer', 15) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 6.38961791992e-05
   0.000140s - ('blas_opt_inplace', 'TopoOptimizer', 22) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.41074371338e-05
       loop time 1.50203704834e-05
       callback_time 0.0
   0.000139s - ('merge3', 'MergeOptimizer', 40) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 0.000133991241455
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000136s - ('InplaceGpuBlasOpt', 'TopoOptimizer', 23) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.69277191162e-05
       callback_time 0.0
   0.000135s - ('gpuablas_opt_inplace', 'TopoOptimizer', 24) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.69277191162e-05
       callback_time 0.0
   0.000130s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 34) - 0.000s
   0.000129s - ('local_dnn_conv_inplace', 'TopoOptimizer', 26) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 1.09672546387e-05
       callback_time 0.0
   0.000107s - ('uncanonicalize', 'EquilibriumOptimizer', 11) - 0.000s
     EquilibriumOptimizer      uncanonicalize
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000102s - ('specialize_device', 'EquilibriumOptimizer', 13) - 0.000s
     EquilibriumOptimizer      specialize_device
       time 0.000s for 1 passes
       nb nodes (start, end,  max) 4 4 4
       time io_toposort 0.000s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.000s 0 (0.000s in global opts, 0.000s io_toposort) - 4 nodes - 
   0.000097s - ('local_inplace_incsubtensor1', 'TopoOptimizer', 20) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.38690185547e-05
       loop time 6.19888305664e-06
       callback_time 0.0
   0.000096s - ('gpu_scanOp_make_inplace', 'ScanInplaceOptimizer', 31) - 0.000s
   0.000096s - ('local_inplace_setsubtensor', 'TopoOptimizer', 21) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.50611114502e-05
       loop time 6.91413879395e-06
       callback_time 0.0
   0.000094s - ('scanOp_make_inplace', 'ScanInplaceOptimizer', 35) - 0.000s
   0.000093s - ('gpua_scanOp_make_inplace', 'ScanInplaceOptimizer', 33) - 0.000s
   0.000092s - ('dimshuffle_as_view', 'TopoOptimizer', 19) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 5.60283660889e-05
       loop time 6.91413879395e-06
       callback_time 0.0
   0.000090s - ('merge2', 'MergeOptimizer', 16) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 8.6784362793e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000090s - ('local_gemm16_inplace', 'TopoOptimizer', 27) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 6.91413879395e-06
       callback_time 0.0
   0.000078s - ('c_blas_destructive', 'TopoOptimizer', 25) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.29153442383e-05
       loop time 1.28746032715e-05
       callback_time 0.0
   0.000078s - ('local_IncSubtensor_serialize', 'TopoOptimizer', 3) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.10079956055e-05
       loop time 1.50203704834e-05
       callback_time 0.0
   0.000072s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.000s
   0.000071s - ('local_destructive', 'TopoOptimizer', 37) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 9.05990600586e-06
       callback_time 0.0
   0.000071s - ('crossentropy_to_crossentropy_with_softmax', 'FromFunctionOptimizer', 10) - 0.000s
   0.000070s - ('mrg_random_make_inplace', 'TopoOptimizer', 39) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 3.88622283936e-05
       loop time 1.00135803223e-05
       callback_time 0.0
   0.000070s - ('cond_make_inplace', 'TopoOptimizer', 36) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 5.96046447754e-06
       callback_time 0.0
   0.000068s - ('make_ger_destructive', 'TopoOptimizer', 28) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 4.00543212891e-05
       loop time 7.86781311035e-06
       callback_time 0.0
   0.000068s - ('random_make_inplace', 'TopoOptimizer', 38) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (4, 4, 0)
       init io_toposort 3.91006469727e-05
       loop time 7.86781311035e-06
       callback_time 0.0
   0.000066s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 32) - 0.000s
   0.000027s - ('merge1.2', 'MergeOptimizer', 5) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 2.31266021729e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0

Function profiling
==================
  Message: /usr/local/lib/python2.7/dist-packages/blocks-0.0.1-py2.7.egg/blocks/monitoring/evaluators.py:275
  Time in 1801 calls to Function.__call__: 1.578782e+01s
  Time in Function.fn.__call__: 1.565544e+01s (99.161%)
  Time in thunks: 1.532817e+01s (97.089%)
  Total compile time: 8.366465e+01s
    Number of Apply nodes: 131
    Theano Optimizer time: 1.449493e+00s
       Theano validate time: 5.916786e-02s
    Theano Linker time (includes C, CUDA code generation/compiling): 8.218922e+01s
       Import time 1.190503e-01s

Time in all call to theano.grad() 6.174994e-02s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  31.1%    31.1%       4.772s       2.65e-03s     C     1801       1   theano.sandbox.cuda.dnn.GpuDnnPool
  17.8%    49.0%       2.732s       7.22e-05s     C    37817      21   theano.sandbox.cuda.basic_ops.GpuElemwise
  15.1%    64.1%       2.319s       4.29e-04s     C     5403       3   theano.sandbox.cuda.dnn.GpuDnnConv
  12.7%    76.7%       1.939s       2.69e-04s     C     7204       4   theano.sandbox.cuda.basic_ops.GpuFromHost
   7.0%    83.8%       1.080s       4.61e-05s     C    23413      13   theano.sandbox.cuda.basic_ops.GpuDimShuffle
   5.6%    89.4%       0.864s       6.86e-05s     C    12607       7   theano.sandbox.cuda.basic_ops.GpuContiguous
   3.9%    93.3%       0.596s       1.10e-04s     C     5403       3   theano.sandbox.cuda.basic_ops.GpuAllocEmpty
   1.5%    94.8%       0.233s       1.29e-05s     Py   18006       4   theano.ifelse.IfElse
   1.4%    96.2%       0.214s       5.93e-05s     C     3602       2   theano.compile.ops.DeepCopyOp
   1.0%    97.2%       0.151s       1.67e-05s     C     9005       5   theano.sandbox.cuda.basic_ops.HostFromGpu
   0.7%    97.9%       0.103s       2.87e-05s     C     3602       2   theano.sandbox.cuda.basic_ops.GpuCAReduce
   0.6%    98.5%       0.091s       2.82e-06s     C    32414      18   theano.tensor.elemwise.Elemwise
   0.6%    99.1%       0.091s       2.52e-05s     C     3602       2   theano.sandbox.cuda.blas.GpuDot22
   0.3%    99.4%       0.048s       1.15e-06s     C    41423      23   theano.compile.ops.Shape_i
   0.2%    99.6%       0.033s       3.09e-06s     C    10806       6   theano.sandbox.cuda.basic_ops.GpuSubtensor
   0.1%    99.7%       0.018s       1.26e-06s     C    14408       8   theano.tensor.opt.MakeVector
   0.1%    99.8%       0.018s       3.27e-06s     C     5403       3   theano.sandbox.cuda.dnn.GpuDnnConvDesc
   0.0%    99.9%       0.008s       4.17e-06s     C     1801       1   theano.tensor.elemwise.DimShuffle
   0.0%    99.9%       0.007s       1.89e-06s     C     3602       2   theano.tensor.elemwise.Sum
   0.0%   100.0%       0.006s       1.77e-06s     C     3602       2   theano.tensor.subtensor.Subtensor
   ... (remaining 1 Classes account for   0.03%(0.01s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  31.1%    31.1%       4.772s       2.65e-03s     C     1801        1   GpuDnnPool
  15.1%    46.3%       2.319s       4.29e-04s     C     5403        3   GpuDnnConv{workmem='small', inplace=True}
  12.7%    58.9%       1.939s       2.69e-04s     C     7204        4   GpuFromHost
   6.3%    65.2%       0.968s       1.34e-04s     C     7204        4   GpuElemwise{Composite{Cast{float32}(GT(i0, i1))},no_inplace}
   5.6%    70.9%       0.864s       6.86e-05s     C     12607        7   GpuContiguous
   4.4%    75.2%       0.671s       9.32e-05s     C     7204        4   GpuElemwise{Switch}[(0, 0)]
   3.9%    79.1%       0.598s       6.64e-05s     C     9005        5   GpuElemwise{Add}[(0, 0)]
   3.9%    83.0%       0.596s       1.10e-04s     C     5403        3   GpuAllocEmpty
   3.5%    86.5%       0.534s       2.96e-04s     C     1801        1   GpuDimShuffle{0,2,1,x}
   2.4%    88.9%       0.368s       6.80e-05s     C     5403        3   GpuDimShuffle{0,1,2,x}
   2.3%    91.2%       0.350s       6.49e-05s     C     5399        3   GpuElemwise{Add}[(0, 1)]
   1.4%    92.6%       0.214s       5.93e-05s     C     3602        2   DeepCopyOp
   1.1%    93.7%       0.166s       1.54e-05s     Py    10806        2   if{inplace}
   1.0%    94.7%       0.151s       1.67e-05s     C     9005        5   HostFromGpu
   0.9%    95.5%       0.133s       7.38e-05s     C     1801        1   GpuDimShuffle{0,2,1}
   0.6%    96.1%       0.091s       2.52e-05s     C     3602        2   GpuDot22
   0.4%    96.6%       0.067s       9.29e-06s     Py    7200        2   if{inplace,gpu}
   0.4%    96.9%       0.059s       3.29e-05s     C     1801        1   GpuCAReduce{add}{1,1}
   0.3%    97.2%       0.044s       2.45e-05s     C     1801        1   GpuCAReduce{add}{0,1,0}
   0.2%    97.4%       0.031s       1.75e-05s     C     1801        1   GpuElemwise{Composite{((i0 / i1) / i2)}}[(0, 0)]
   ... (remaining 35 Ops account for   2.57%(0.39s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  31.1%    31.1%       4.772s       2.65e-03s   1801    82   GpuDnnPool(GpuContiguous.0, GpuDnnPoolDesc{ws=(5, 1), stride=(1, 1), mode='max', pad=(0, 0)}.0)
  11.1%    42.2%       1.697s       9.42e-04s   1801    12   GpuFromHost(features)
   6.2%    48.4%       0.952s       5.29e-04s   1801    87   GpuDnnConv{workmem='small', inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(3, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   4.8%    53.2%       0.737s       4.09e-04s   1801    71   GpuDnnConv{workmem='small', inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(1, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   4.1%    57.3%       0.631s       3.50e-04s   1801   101   GpuDnnConv{workmem='small', inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(3, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   3.5%    60.8%       0.534s       2.96e-04s   1801    40   GpuDimShuffle{0,2,1,x}(GpuFromHost.0)
   3.4%    64.2%       0.521s       2.89e-04s   1801    69   GpuContiguous(GpuElemwise{Add}[(0, 1)].0)
   2.8%    67.0%       0.427s       2.37e-04s   1801    79   GpuElemwise{Composite{Cast{float32}(GT(i0, i1))},no_inplace}(GpuElemwise{Add}[(0, 0)].0, CudaNdarrayConstant{[[[[ 0.]]]]})
   2.1%    69.1%       0.320s       1.78e-04s   1801    78   GpuElemwise{Add}[(0, 0)](GpuDimShuffle{0,1,2,x}.0, GpuDimShuffle{x,0,x,x}.0)
   2.1%    71.2%       0.318s       1.77e-04s   1801    80   GpuElemwise{Switch}[(0, 0)](GpuElemwise{Composite{Cast{float32}(GT(i0, i1))},no_inplace}.0, GpuElemwise{Add}[(0, 0)].0, CudaNdarrayConstant{[[[[ 0.]]]]})
   1.9%    73.1%       0.292s       1.62e-04s   1801    68   GpuElemwise{Add}[(0, 1)](GpuElemwise{Composite{((i0 / i1) / i2)}}[(0, 0)].0, GpuDimShuffle{0,2,1,x}.0)
   1.9%    74.9%       0.287s       1.59e-04s   1801    97   GpuElemwise{Composite{Cast{float32}(GT(i0, i1))},no_inplace}(GpuElemwise{Add}[(0, 0)].0, CudaNdarrayConstant{[[[[ 0.]]]]})
   1.5%    76.4%       0.228s       1.27e-04s   1801   105   GpuElemwise{Composite{Cast{float32}(GT(i0, i1))},no_inplace}(GpuElemwise{Add}[(0, 0)].0, CudaNdarrayConstant{[[[ 0.]]]})
   1.4%    77.9%       0.219s       1.22e-04s   1801    77   GpuAllocEmpty(Shape_i{0}.0, Shape_i{0}.0, Elemwise{Composite{(i0 + ((i1 - i2) // i3))}}[(0, 1)].0, Elemwise{Composite{(i0 + ((i0 - i1) // i0))}}[(0, 1)].0)
   1.3%    79.2%       0.206s       1.14e-04s   1801   106   GpuElemwise{Switch}[(0, 0)](GpuElemwise{Composite{Cast{float32}(GT(i0, i1))},no_inplace}.0, GpuElemwise{Add}[(0, 0)].0, CudaNdarrayConstant{[[[ 0.]]]})
   1.3%    80.5%       0.205s       1.14e-04s   1801    62   GpuAllocEmpty(Shape_i{0}.0, Shape_i{0}.0, Elemwise{add,no_inplace}.0, Elemwise{Composite{(i0 + ((i0 - i1) // i0))}}[(0, 1)].0)
   1.2%    81.7%       0.184s       1.02e-04s   1801    92   GpuDimShuffle{0,1,2,x}(GpuSubtensor{::, ::, ::, int64}.0)
   1.2%    82.9%       0.181s       1.01e-04s   1801    76   GpuDimShuffle{0,1,2,x}(GpuSubtensor{::, ::, ::, int64}.0)
   1.1%    84.0%       0.172s       9.53e-05s   1801    94   GpuAllocEmpty(Shape_i{0}.0, Shape_i{0}.0, Elemwise{Composite{(i0 + ((i1 - i2) // i3))}}[(0, 1)].0, Elemwise{Composite{(i0 + ((i0 - i1) // i0))}}[(0, 1)].0)
   1.1%    85.1%       0.170s       9.44e-05s   1801    81   GpuContiguous(GpuElemwise{Switch}[(0, 0)].0)
   ... (remaining 111 Apply instances account for 14.85%(2.28s) of the runtime)

Optimizer Profile
-----------------
 SeqOptimizer  time 1.449s for 217/129 nodes before/after optimization
   0.059s for fgraph.validate()
   0.206s for callback
   time      - (name, class, index) - validate time
   0.809661s - ('gpu_opt', 'SeqOptimizer', 12) - 0.001s
     SeqOptimizer      gpu_opt  time 0.810s for 165/165 nodes before/after optimization
       0.001s for fgraph.validate()
       0.040s for callback
       0.791890s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.001s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.792s for 5 passes
           nb nodes (start, end,  max) 169 229 230
           time io_toposort 0.009s
           time in local optimizers 0.769s
           time in global optimizers 0.003s
            0 - 0.041s 33 (0.001s in global opts, 0.001s io_toposort) - 169 nodes - ('local_gpu_dimshuffle_0', 10) ('local_gpu_subtensor', 6) ('local_gpu_elemwise_1', 6) ('constant_folding', 4) ('local_gpu_careduce', 2) ...
            1 - 0.015s 9 (0.001s in global opts, 0.002s io_toposort) - 192 nodes - ('local_gpu_elemwise_0', 5) ('local_gpu_dimshuffle_0', 3) ('MergeOptimizer', 1)
            2 - 0.585s 11 (0.000s in global opts, 0.002s io_toposort) - 207 nodes - ('local_gpu_elemwise_0', 7) ('constant_folding', 4)
            3 - 0.143s 5 (0.001s in global opts, 0.002s io_toposort) - 230 nodes - ('constant_folding', 3) ('MergeOptimizer', 1) ('local_gpu_elemwise_0', 1)
            4 - 0.008s 0 (0.000s in global opts, 0.002s io_toposort) - 229 nodes - 
           times - times applied - nb node created - name:
           0.705s - 11 - 0 - constant_folding
           0.019s - 13 - 60 - local_gpu_elemwise_0
           0.011s - 13 - 31 - local_gpu_dimshuffle_0
           0.008s - 6 - 18 - local_gpu_elemwise_1
           0.005s - 2 - 6 - local_gpu_careduce
           0.004s - 6 - 12 - local_gpu_subtensor
           0.003s - 3 - 0 - MergeOptimizer
           0.002s - 2 - 8 - local_gpu_dot22
           0.002s - 2 - 6 - local_gpu_lazy_ifelse
           0.012s - in 55 optimization that where not used (display only those with a runtime > 0)
             0.002s - local_track_shape_i
             0.001s - local_gpu_dot_to_dot22
             0.001s - local_gpu_ger
             0.001s - local_elemwise_alloc
             0.001s - local_gpu_gemv
             0.001s - local_gpu_conv
             0.000s - local_gpu_specifyShape_0
             0.000s - local_dnn_conv_output_merge
             0.000s - gpuScanOptimization
             0.000s - local_gpu_dot22scalar
             0.000s - local_gpu_solve
             0.000s - local_gpu_gemm
             0.000s - local_gpu_eye
             0.000s - local_gpu_reshape
             0.000s - local_gpu_flatten
             0.000s - local_gpu_incsubtensor
             0.000s - local_gpu_advanced_subtensor1
             0.000s - local_gpu_advanced_incsubtensor1
             0.000s - local_gpu_allocempty
             0.000s - local_dnn_convw_output_merge
             0.000s - local_dnn_convi_output_merge
             0.000s - local_dnn_conv_alpha_merge
             0.000s - local_dnn_convi_alpha_merge
             0.000s - local_dnn_convw_alpha_merge
             0.000s - local_gpu_contiguous_gpu_contiguous
             0.000s - local_subtensor_make_vector
             0.000s - local_gpu_elemwise_careduce

       0.016688s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.017s for 2 passes
           nb nodes (start, end,  max) 229 165 229
           time io_toposort 0.004s
           time in local optimizers 0.011s
           time in global optimizers 0.000s
            0 - 0.014s 36 (0.000s in global opts, 0.002s io_toposort) - 229 nodes - ('local_cut_gpu_host_gpu', 36)
            1 - 0.003s 0 (0.000s in global opts, 0.002s io_toposort) - 165 nodes - 
           times - times applied - nb node created - name:
           0.011s - 36 - 0 - local_cut_gpu_host_gpu
           0.001s - in 1 optimization that where not used (display only those with a runtime > 0)
             0.001s - constant_folding

       0.001056s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000007s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.234304s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.003s
     EquilibriumOptimizer      canonicalize
       time 0.234s for 7 passes
       nb nodes (start, end,  max) 199 165 199
       time io_toposort 0.012s
       time in local optimizers 0.189s
       time in global optimizers 0.019s
        0 - 0.063s 61 (0.000s in global opts, 0.002s io_toposort) - 199 nodes - ('local_useless_elemwise', 18) ('local_upcast_elemwise_constant_inputs', 14) ('local_shape_to_shape_i', 10) ('local_dimshuffle_lift', 8) ('local_add_canonizer', 6) ...
        1 - 0.049s 39 (0.014s in global opts, 0.002s io_toposort) - 199 nodes - ('local_subtensor_make_vector', 16) ('local_dimshuffle_lift', 10) ('local_add_canonizer', 4) ('local_useless_elemwise', 4) ('local_upcast_elemwise_constant_inputs', 4) ...
        2 - 0.041s 15 (0.002s in global opts, 0.002s io_toposort) - 185 nodes - ('local_add_canonizer', 6) ('local_dimshuffle_lift', 4) ('constant_folding', 4) ('MergeOptimizer', 1)
        3 - 0.031s 11 (0.002s in global opts, 0.002s io_toposort) - 176 nodes - ('constant_folding', 6) ('local_dimshuffle_lift', 4) ('MergeOptimizer', 1)
        4 - 0.022s 8 (0.001s in global opts, 0.002s io_toposort) - 172 nodes - ('local_dimshuffle_lift', 6) ('MergeOptimizer', 1) ('constant_folding', 1)
        5 - 0.014s 1 (0.000s in global opts, 0.001s io_toposort) - 165 nodes - ('MergeOptimizer', 1)
        6 - 0.014s 0 (0.000s in global opts, 0.001s io_toposort) - 165 nodes - 
       times - times applied - nb node created - name:
       0.038s - 16 - 18 - local_add_canonizer
       0.036s - 14 - 0 - constant_folding
       0.032s - 32 - 66 - local_dimshuffle_lift
       0.020s - 18 - 54 - local_upcast_elemwise_constant_inputs
       0.019s - 5 - 2 - MergeOptimizer
       0.007s - 22 - 0 - local_useless_elemwise
       0.007s - 10 - 39 - local_shape_to_shape_i
       0.003s - 16 - 0 - local_subtensor_make_vector
       0.001s - 1 - 2 - local_subtensor_lift
       0.001s - 1 - 1 - local_neg_to_mul
       0.044s - in 67 optimization that where not used (display only those with a runtime > 0)
         0.006s - local_one_minus_erf2
         0.006s - local_mul_canonizer
         0.005s - local_one_minus_erf
         0.004s - local_greedy_distributor
         0.003s - local_func_inv
         0.003s - local_fill_sink
         0.003s - local_track_shape_i
         0.002s - local_cut_gpu_host_gpu
         0.002s - local_useless_subtensor
         0.001s - local_remove_switch_const_cond
         0.001s - local_fill_cut
         0.001s - local_cast_cast
         0.001s - local_mul_zero
         0.001s - local_useless_slice
         0.001s - local_IncSubtensor_serialize
         0.001s - local_div_switch_sink
         0.000s - local_0_dot_x
         0.000s - local_lift_transpose_through_dot
         0.000s - local_mul_switch_sink
         0.000s - local_subtensor_of_alloc
         0.000s - local_subtensor_merge
         0.000s - local_sum_div_dimshuffle
         0.000s - local_subtensor_of_dot
         0.000s - local_dimshuffle_no_inplace_at_canonicalize
         0.000s - local_op_of_op
         0.000s - local_cut_useless_reduce
         0.000s - local_sum_prod_all_to_none
         0.000s - local_reduce_join

   0.063930s - ('gpu_after_fusion', 'SeqOptimizer', 17) - 0.000s
     SeqOptimizer      gpu_after_fusion  time 0.064s for 137/129 nodes before/after optimization
       0.000s for fgraph.validate()
       0.014s for callback
       0.054820s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.055s for 4 passes
           nb nodes (start, end,  max) 141 149 152
           time io_toposort 0.005s
           time in local optimizers 0.039s
           time in global optimizers 0.005s
            0 - 0.026s 8 (0.000s in global opts, 0.001s io_toposort) - 141 nodes - ('local_gpu_elemwise_0', 4) ('constant_folding', 4)
            1 - 0.014s 4 (0.001s in global opts, 0.001s io_toposort) - 152 nodes - ('constant_folding', 3) ('MergeOptimizer', 1)
            2 - 0.009s 1 (0.004s in global opts, 0.001s io_toposort) - 149 nodes - ('MergeOptimizer', 1)
            3 - 0.006s 0 (0.000s in global opts, 0.001s io_toposort) - 149 nodes - 
           times - times applied - nb node created - name:
           0.017s - 4 - 20 - local_gpu_elemwise_0
           0.014s - 7 - 0 - constant_folding
           0.005s - 2 - 0 - MergeOptimizer
           0.008s - in 61 optimization that where not used (display only those with a runtime > 0)
             0.001s - local_track_shape_i
             0.001s - local_gpu_elemwise_1
             0.001s - local_elemwise_alloc
             0.000s - local_dnn_conv_output_merge
             0.000s - local_gpu_lazy_ifelse
             0.000s - local_dnn_conv_alpha_merge
             0.000s - local_dnn_convw_output_merge
             0.000s - local_dnn_convi_output_merge
             0.000s - local_gpu_dot_to_dot22
             0.000s - local_gpu_gemv
             0.000s - local_gpu_ger
             0.000s - local_gpu_dimshuffle_0
             0.000s - local_gpu_conv
             0.000s - local_dnn_convw_alpha_merge
             0.000s - local_dnn_convi_alpha_merge
             0.000s - local_gpu_dot22
             0.000s - local_gpu_subtensor
             0.000s - local_gpu_dot22scalar
             0.000s - local_gpu_specifyShape_0
             0.000s - gpuScanOptimization
             0.000s - local_gpu_solve
             0.000s - local_gpu_gemm
             0.000s - local_gpu_eye
             0.000s - local_gpu_reshape
             0.000s - local_gpu_incsubtensor
             0.000s - local_gpu_flatten
             0.000s - local_gpu_advanced_incsubtensor1
             0.000s - local_gpu_advanced_subtensor1
             0.000s - local_gpu_allocempty
             0.000s - local_gpu_contiguous_gpu_contiguous
             0.000s - local_gpu_careduce
             0.000s - local_subtensor_make_vector
             0.000s - local_gpu_elemwise_careduce

       0.008024s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.008s for 2 passes
           nb nodes (start, end,  max) 149 129 149
           time io_toposort 0.002s
           time in local optimizers 0.004s
           time in global optimizers 0.000s
            0 - 0.006s 12 (0.000s in global opts, 0.001s io_toposort) - 149 nodes - ('local_cut_gpu_host_gpu', 12)
            1 - 0.002s 0 (0.000s in global opts, 0.001s io_toposort) - 129 nodes - 
           times - times applied - nb node created - name:
           0.004s - 12 - 0 - local_cut_gpu_host_gpu
           0.001s - in 1 optimization that where not used (display only those with a runtime > 0)
             0.001s - constant_folding

       0.001058s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000013s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.060028s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 34) - 0.019s
   0.050266s - ('elemwise_fusion', 'SeqOptimizer', 14) - 0.000s
     SeqOptimizer      elemwise_fusion  time 0.050s for 165/139 nodes before/after optimization
       0.000s for fgraph.validate()
       0.006s for callback
       0.048613s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
         FusionOptimizer
          nb_iter 2
          nb_replacement 15
          nb_inconsistency_replace 0
          validate_time 0.000201940536499
          callback_time 0.00552821159363
          time_toposort 0.00385093688965
       0.001642s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 0.00135493278503

   0.042228s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.016s
   0.038169s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
     EquilibriumOptimizer      specialize
       time 0.038s for 3 passes
       nb nodes (start, end,  max) 167 165 167
       time io_toposort 0.004s
       time in local optimizers 0.024s
       time in global optimizers 0.005s
        0 - 0.013s 3 (0.001s in global opts, 0.001s io_toposort) - 167 nodes - ('local_mul_specialize', 3)
        1 - 0.014s 2 (0.002s in global opts, 0.001s io_toposort) - 167 nodes - ('constant_folding', 2)
        2 - 0.011s 0 (0.002s in global opts, 0.001s io_toposort) - 165 nodes - 
       times - times applied - nb node created - name:
       0.004s - 2 - 0 - constant_folding
       0.003s - 3 - 5 - local_mul_specialize
       0.023s - in 63 optimization that where not used (display only those with a runtime > 0)
         0.005s - crossentropy_to_crossentropy_with_softmax_with_bias
         0.004s - local_add_specialize
         0.004s - local_one_minus_erf2
         0.002s - local_one_minus_erf
         0.001s - local_func_inv
         0.001s - local_elemwise_alloc
         0.001s - local_useless_elemwise
         0.001s - local_track_shape_i
         0.001s - local_div_to_inv
         0.000s - local_useless_subtensor
         0.000s - local_elemwise_sub_zeros
         0.000s - local_cast_cast
         0.000s - local_alloc_unary
         0.000s - local_useless_slice
         0.000s - local_abs_merge
         0.000s - local_grad_log_erfc_neg
         0.000s - local_dimshuffle_lift
         0.000s - local_subtensor_make_vector
         0.000s - local_sum_div_dimshuffle
         0.000s - local_neg_div_neg
         0.000s - local_reduce_broadcastable
         0.000s - local_sum_prod_mul_by_scalar
         0.000s - local_subtensor_merge
         0.000s - local_mul_to_sqr
         0.000s - local_subtensor_of_dot
         0.000s - local_subtensor_of_alloc
         0.000s - local_opt_alloc
         0.000s - local_neg_neg

   0.014998s - ('stabilize', 'EquilibriumOptimizer', 6) - 0.000s
     EquilibriumOptimizer      stabilize
       time 0.015s for 2 passes
       nb nodes (start, end,  max) 165 167 167
       time io_toposort 0.002s
       time in local optimizers 0.008s
       time in global optimizers 0.003s
        0 - 0.008s 2 (0.002s in global opts, 0.001s io_toposort) - 165 nodes - ('Elemwise{log,no_inplace}(Elemwise{sub,no_inplace}(y subject to <function _is_1 at 0x7fb67a153b90>, sigmoid(x))) -> Elemwise{neg,no_inplace}(softplus(x))', 1) ('Elemwise{log,no_inplace}(sigmoid(x)) -> Elemwise{neg,no_inplace}(softplus(Elemwise{neg,no_inplace}(x)))', 1)
        1 - 0.007s 0 (0.001s in global opts, 0.001s io_toposort) - 167 nodes - 
       times - times applied - nb node created - name:
       0.001s - 1 - 2 - Elemwise{log,no_inplace}(Elemwise{sub,no_inplace}(y subject to <function _is_1 at 0x7fb67a153b90>, sigmoid(x))) -> Elemwise{neg,no_inplace}(softplus(x))
       0.001s - 1 - 3 - Elemwise{log,no_inplace}(sigmoid(x)) -> Elemwise{neg,no_inplace}(softplus(Elemwise{neg,no_inplace}(x)))
       0.009s - in 34 optimization that where not used (display only those with a runtime > 0)
         0.003s - crossentropy_to_crossentropy_with_softmax_with_bias
         0.002s - local_one_minus_erf2
         0.001s - local_greedy_distributor
         0.001s - local_one_minus_erf
         0.001s - constant_folding
         0.000s - local_sigm_times_exp
         0.000s - local_exp_over_1_plus_exp
         0.000s - local_0_dot_x
         0.000s - local_grad_log_erfc_neg
         0.000s - local_subtensor_of_dot
         0.000s - local_log_erfc
         0.000s - local_log1p
         0.000s - local_log_add

   0.013195s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
     SeqOptimizer      BlasOpt  time 0.013s for 167/167 nodes before/after optimization
       0.000s for fgraph.validate()
       0.001s for callback
       0.002847s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (167, 167, 2)
           init io_toposort 0.00136494636536
           loop time 0.00142884254456
           callback_time 0.000795125961304
       0.002670s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
         GemmOptimizer
          nb_iter 1
          nb_replacement 0
          nb_replacement_didn_t_remove 0
          nb_inconsistency_make 0
          nb_inconsistency_replace 0
          time_canonicalize 0.00031852722168
          time_factor_can 0
          time_factor_list 0
          time_toposort 0.00141191482544
          validate_time 0.0
          callback_time 0.0
       0.002250s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (167, 167, 0)
           init io_toposort 0.00184607505798
           loop time 0.000345945358276
           callback_time 0.0
       0.002027s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          local_gemm_to_gemv
           time 0.002s for 1 passes
           nb nodes (start, end,  max) 167 167 167
           time io_toposort 0.001s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.002s 0 (0.000s in global opts, 0.001s io_toposort) - 167 nodes - 
       0.001694s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (167, 167, 0)
           init io_toposort 0.00139808654785
           loop time 0.000241041183472
           callback_time 0.0
       0.001681s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (167, 167, 0)
           init io_toposort 0.00139307975769
           loop time 0.000229120254517
           callback_time 0.0

   0.012892s - ('scan_eqopt2', 'EquilibriumOptimizer', 7) - 0.000s
     EquilibriumOptimizer      scan_eqopt2
       time 0.013s for 1 passes
       nb nodes (start, end,  max) 167 167 167
       time io_toposort 0.001s
       time in local optimizers 0.000s
       time in global optimizers 0.011s
        0 - 0.013s 0 (0.011s in global opts, 0.001s io_toposort) - 167 nodes - 
   0.012230s - ('ShapeOpt', 'ShapeOptimizer', 1) - 0.000s
   0.011967s - ('cond_make_inplace', 'TopoOptimizer', 36) - 0.009s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 4)
       init io_toposort 0.00101900100708
       loop time 0.0108749866486
       callback_time 0.00959372520447
   0.011361s - ('scan_eqopt1', 'EquilibriumOptimizer', 2) - 0.000s
     EquilibriumOptimizer      scan_eqopt1
       time 0.011s for 1 passes
       nb nodes (start, end,  max) 199 199 199
       time io_toposort 0.002s
       time in local optimizers 0.000s
       time in global optimizers 0.009s
        0 - 0.011s 0 (0.009s in global opts, 0.002s io_toposort) - 199 nodes - 
   0.009256s - ('add_destroy_handler', 'AddDestroyHandler', 18) - 0.000s
   0.008332s - ('merge3', 'MergeOptimizer', 40) - 0.007s
     MergeOptimizer
       nb_fail 0
       replace_time 0.00831890106201
       validate_time 0.00713300704956
       callback_time 0.00720834732056
       nb_merged 4
       nb_constant 4
   0.008236s - ('merge1', 'MergeOptimizer', 0) - 0.001s
     MergeOptimizer
       nb_fail 0
       replace_time 0.00441598892212
       validate_time 0.000510215759277
       callback_time 0.00162315368652
       nb_merged 48
       nb_constant 30
   0.008095s - ('gpu_elemwise_fusion', 'FusionOptimizer', 15) - 0.000s
     FusionOptimizer
      nb_iter 2
      nb_replacement 2
      nb_inconsistency_replace 0
      validate_time 2.88486480713e-05
      callback_time 0.000923156738281
      time_toposort 0.00271487236023
   0.006582s - ('local_dnn_conv_inplace', 'TopoOptimizer', 26) - 0.002s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 3)
       init io_toposort 0.000982046127319
       loop time 0.00547409057617
       callback_time 0.0030243396759
   0.002908s - ('local_IncSubtensor_serialize', 'TopoOptimizer', 3) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (199, 199, 0)
       init io_toposort 0.00236701965332
       loop time 0.000480890274048
       callback_time 0.0
   0.002372s - ('scanOp_make_inplace', 'ScanInplaceOptimizer', 35) - 0.000s
   0.002237s - ('dimshuffle_as_view', 'TopoOptimizer', 19) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 1)
       init io_toposort 0.00110507011414
       loop time 0.00107407569885
       callback_time 0.000616788864136
   0.001989s - ('gpu_scanOp_make_inplace', 'ScanInplaceOptimizer', 31) - 0.000s
   0.001986s - ('gpua_scanOp_make_inplace', 'ScanInplaceOptimizer', 33) - 0.000s
   0.001854s - ('uncanonicalize', 'EquilibriumOptimizer', 11) - 0.000s
     EquilibriumOptimizer      uncanonicalize
       time 0.002s for 1 passes
       nb nodes (start, end,  max) 165 165 165
       time io_toposort 0.001s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.002s 0 (0.000s in global opts, 0.001s io_toposort) - 165 nodes - 
   0.001666s - ('specialize_device', 'EquilibriumOptimizer', 13) - 0.000s
     EquilibriumOptimizer      specialize_device
       time 0.002s for 1 passes
       nb nodes (start, end,  max) 165 165 165
       time io_toposort 0.001s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.002s 0 (0.000s in global opts, 0.001s io_toposort) - 165 nodes - 
   0.001589s - ('crossentropy_to_crossentropy_with_softmax', 'FromFunctionOptimizer', 10) - 0.000s
   0.001539s - ('blas_opt_inplace', 'TopoOptimizer', 22) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00104689598083
       loop time 0.000349998474121
       callback_time 0.0
   0.001497s - ('InplaceGpuBlasOpt', 'TopoOptimizer', 23) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00104212760925
       loop time 0.000322103500366
       callback_time 0.0
   0.001398s - ('gpuablas_opt_inplace', 'TopoOptimizer', 24) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.000988006591797
       loop time 0.000294923782349
       callback_time 0.0
   0.001353s - ('gpua_elemwise_fusion', 'FusionOptimizer', 29) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 0.00108981132507
   0.001325s - ('local_inplace_incsubtensor1', 'TopoOptimizer', 20) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00111603736877
       loop time 0.000128984451294
       callback_time 0.0
   0.001293s - ('local_gemm16_inplace', 'TopoOptimizer', 27) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00112700462341
       loop time 9.60826873779e-05
       callback_time 0.0
   0.001260s - ('c_blas_destructive', 'TopoOptimizer', 25) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00095796585083
       loop time 0.000247001647949
       callback_time 0.0
   0.001252s - ('local_inplace_setsubtensor', 'TopoOptimizer', 21) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00102806091309
       loop time 0.000123023986816
       callback_time 0.0
   0.001241s - ('mrg_random_make_inplace', 'TopoOptimizer', 39) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00103306770325
       loop time 0.000156879425049
       callback_time 0.0
   0.001208s - ('local_destructive', 'TopoOptimizer', 37) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00102496147156
       loop time 0.000128030776978
       callback_time 0.0
   0.001181s - ('random_make_inplace', 'TopoOptimizer', 38) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.00100588798523
       loop time 0.000127077102661
       callback_time 0.0
   0.001167s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 32) - 0.000s
   0.001123s - ('make_ger_destructive', 'TopoOptimizer', 28) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (129, 129, 0)
       init io_toposort 0.000945091247559
       loop time 0.000147104263306
       callback_time 0.0
   0.000132s - ('merge2', 'MergeOptimizer', 16) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 0.000124216079712
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000051s - ('merge1.2', 'MergeOptimizer', 5) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 4.50611114502e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0

Function profiling
==================
  Message: /usr/local/lib/python2.7/dist-packages/blocks-0.0.1-py2.7.egg/blocks/algorithms/__init__.py:224
  Time in 1563 calls to Function.__call__: 6.121255e+01s
  Time in Function.fn.__call__: 6.108444e+01s (99.791%)
  Time in thunks: 6.018744e+01s (98.325%)
  Total compile time: 9.430349e+01s
    Number of Apply nodes: 314
    Theano Optimizer time: 4.506236e+00s
       Theano validate time: 3.787827e-01s
    Theano Linker time (includes C, CUDA code generation/compiling): 8.966871e+01s
       Import time 1.939845e-02s

Time in all call to theano.grad() 6.174994e-02s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  33.3%    33.3%      20.027s       4.75e-04s     C    42199      27   theano.tensor.elemwise.Elemwise
  13.3%    46.5%       7.984s       5.11e-04s     C    15630      10   theano.sandbox.cuda.basic_ops.GpuFromHost
   7.9%    54.5%       4.781s       2.91e-05s     C   164113     105   theano.sandbox.cuda.basic_ops.GpuElemwise
   7.7%    62.2%       4.655s       3.72e-04s     C    12504       8   theano.sandbox.cuda.basic_ops.HostFromGpu
   7.1%    69.4%       4.301s       2.75e-03s     C     1563       1   theano.sandbox.cuda.dnn.GpuDnnPool
   6.2%    75.6%       3.751s       2.40e-03s     C     1563       1   theano.sandbox.cuda.dnn.GpuDnnPoolGrad
   4.1%    79.7%       2.463s       3.94e-05s     C    62520      40   theano.sandbox.cuda.basic_ops.GpuDimShuffle
   3.8%    83.4%       2.259s       7.23e-04s     C     3126       2   theano.sandbox.cuda.dnn.GpuDnnConvGradI
   3.3%    86.7%       1.959s       4.18e-04s     C     4689       3   theano.sandbox.cuda.dnn.GpuDnnConvGradW
   2.7%    89.4%       1.645s       3.51e-04s     C     4689       3   theano.sandbox.cuda.dnn.GpuDnnConv
   2.3%    91.8%       1.408s       8.19e-05s     C    17193      11   theano.sandbox.cuda.basic_ops.GpuContiguous
   1.7%    93.5%       1.023s       6.55e-05s     C    15630      10   theano.sandbox.cuda.basic_ops.GpuAllocEmpty
   1.7%    95.1%       1.002s       1.60e-04s     C     6252       4   theano.sandbox.cuda.basic_ops.GpuAlloc
   1.6%    96.7%       0.953s       1.52e-04s     C     6252       4   theano.sandbox.cuda.basic_ops.GpuIncSubtensor
   1.3%    98.1%       0.811s       6.49e-05s     C    12504       8   theano.sandbox.cuda.basic_ops.GpuCAReduce
   0.4%    98.4%       0.225s       1.44e-04s     C     1563       1   theano.sandbox.cuda.basic_ops.GpuJoin
   0.4%    98.8%       0.216s       2.30e-05s     C     9378       6   theano.sandbox.cuda.blas.GpuDot22
   0.3%    99.1%       0.182s       1.17e-05s     Py   15628       4   theano.ifelse.IfElse
   0.3%    99.4%       0.168s       5.37e-05s     C     3126       2   theano.compile.ops.DeepCopyOp
   0.2%    99.6%       0.115s       7.36e-06s     Py   15630      10   theano.sandbox.cuda.basic_ops.GpuFlatten
   ... (remaining 9 Classes account for   0.43%(0.26s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  22.2%    22.2%      13.390s       1.43e-03s     C     9378        6   Elemwise{Cast{float32}}
  13.3%    35.5%       7.984s       5.11e-04s     C     15630       10   GpuFromHost
  10.9%    46.4%       6.547s       2.09e-03s     C     3126        2   Elemwise{gt,no_inplace}
   7.7%    54.1%       4.655s       3.72e-04s     C     12504        8   HostFromGpu
   7.1%    61.3%       4.301s       2.75e-03s     C     1563        1   GpuDnnPool
   6.2%    67.5%       3.751s       2.40e-03s     C     1563        1   GpuDnnPoolGrad
   3.8%    71.3%       2.259s       7.23e-04s     C     3126        2   GpuDnnConvGradI{inplace=True}
   3.3%    74.5%       1.959s       4.18e-04s     C     4689        3   GpuDnnConvGradW{inplace=True}
   2.7%    77.2%       1.645s       3.51e-04s     C     4689        3   GpuDnnConv{workmem='small', inplace=True}
   2.5%    79.7%       1.481s       3.16e-04s     C     4689        3   GpuDimShuffle{0,2,1,x}
   2.3%    82.0%       1.408s       8.19e-05s     C     17193       11   GpuContiguous
   1.9%    83.9%       1.146s       9.17e-05s     C     12504        8   GpuElemwise{Add}[(0, 0)]
   1.8%    85.7%       1.058s       1.13e-04s     C     9378        6   GpuElemwise{Switch}[(0, 0)]
   1.7%    87.4%       1.023s       6.55e-05s     C     15630       10   GpuAllocEmpty
   1.7%    89.1%       1.002s       1.60e-04s     C     6252        4   GpuAlloc{memset_0=True}
   1.6%    90.7%       0.953s       1.52e-04s     C     6252        4   GpuIncSubtensor{InplaceInc;::, ::, ::, int64}
   0.9%    91.5%       0.532s       5.67e-05s     C     9378        6   GpuDimShuffle{0,2,1}
   0.8%    92.3%       0.488s       2.40e-05s     C     20317       13   GpuElemwise{Add}[(0, 1)]
   0.6%    93.0%       0.367s       1.17e-05s     C     31260       20   GpuElemwise{Composite{(i0 - (i1 ** i2))},no_inplace}
   0.6%    93.5%       0.355s       7.58e-05s     C     4689        3   GpuCAReduce{add}{1,1,0}
   ... (remaining 70 Ops account for   6.45%(3.88s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
   7.5%     7.5%       4.537s       2.90e-03s   1563   145   Elemwise{Cast{float32}}(Elemwise{gt,no_inplace}.0)
   7.1%    14.7%       4.301s       2.75e-03s   1563   151   GpuDnnPool(GpuContiguous.0, GpuDnnPoolDesc{ws=(5, 1), stride=(1, 1), mode='max', pad=(0, 0)}.0)
   6.4%    21.0%       3.827s       2.45e-03s   1563   143   Elemwise{gt,no_inplace}(conv1d_apply_output, TensorConstant{(1, 1, 1) of 0})
   6.3%    27.3%       3.788s       2.42e-03s   1563   174   Elemwise{Cast{float32}}(Elemwise{gt,no_inplace}.0)
   6.2%    33.6%       3.752s       2.40e-03s   1563   146   Elemwise{Cast{float32}}(InplaceDimShuffle{0,2,1,x}.0)
   6.2%    39.8%       3.751s       2.40e-03s   1563   250   GpuDnnPoolGrad(GpuContiguous.0, GpuDnnPool.0, GpuContiguous.0, GpuDnnPoolDesc{ws=(5, 1), stride=(1, 1), mode='max', pad=(0, 0)}.0)
   5.0%    44.8%       3.012s       1.93e-03s   1563   142   HostFromGpu(GpuElemwise{Add}[(0, 0)].0)
   4.5%    49.3%       2.720s       1.74e-03s   1563   172   Elemwise{gt,no_inplace}(conv2_apply_output, TensorConstant{(1, 1, 1) of 0})
   3.7%    53.0%       2.218s       1.42e-03s   1563   147   GpuFromHost(Elemwise{Cast{float32}}.0)
   3.6%    56.6%       2.139s       1.37e-03s   1563   148   GpuFromHost(Elemwise{Cast{float32}}.0)
   2.7%    59.3%       1.641s       1.05e-03s   1563    12   GpuFromHost(features)
   2.7%    62.0%       1.602s       1.02e-03s   1563   244   GpuDnnConvGradI{inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(3, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   2.5%    64.4%       1.498s       9.58e-04s   1563   169   HostFromGpu(GpuElemwise{Add}[(0, 0)].0)
   2.2%    66.6%       1.299s       8.31e-04s   1563   175   Elemwise{Cast{float32}}(InplaceDimShuffle{0,2,1,x}.0)
   1.5%    68.1%       0.927s       5.93e-04s   1563   258   GpuDnnConvGradW{inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(1, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   1.5%    69.6%       0.900s       5.76e-04s   1563   176   GpuFromHost(Elemwise{Cast{float32}}.0)
   1.3%    70.9%       0.754s       4.82e-04s   1563   177   GpuFromHost(Elemwise{Cast{float32}}.0)
   1.2%    72.1%       0.713s       4.56e-04s   1563   140   GpuElemwise{Add}[(0, 0)](GpuDimShuffle{0,2,1}.0, GpuDimShuffle{x,x,0}.0)
   1.2%    73.2%       0.700s       4.48e-04s   1563   245   GpuDnnConvGradW{inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(3, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   1.1%    74.3%       0.657s       4.20e-04s   1563   234   GpuDnnConvGradI{inplace=True}(GpuContiguous.0, GpuContiguous.0, GpuAllocEmpty.0, GpuDnnConvDesc{border_mode='valid', subsample=(3, 1), conv_mode='conv'}.0, Constant{1.0}, Constant{0.0})
   ... (remaining 294 Apply instances account for 25.68%(15.45s) of the runtime)

Optimizer Profile
-----------------
 SeqOptimizer  time 4.506s for 809/312 nodes before/after optimization
   0.379s for fgraph.validate()
   0.989s for callback
   time      - (name, class, index) - validate time
   1.674812s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.007s
     EquilibriumOptimizer      canonicalize
       time 1.675s for 9 passes
       nb nodes (start, end,  max) 684 545 684
       time io_toposort 0.041s
       time in local optimizers 1.483s
       time in global optimizers 0.093s
        0 - 0.513s 163 (0.000s in global opts, 0.005s io_toposort) - 684 nodes - ('local_dimshuffle_lift', 45) ('local_add_canonizer', 26) ('constant_folding', 23) ('local_useless_elemwise', 18) ('local_upcast_elemwise_constant_inputs', 14) ...
        1 - 0.250s 125 (0.076s in global opts, 0.006s io_toposort) - 648 nodes - ('local_dimshuffle_lift', 42) ('local_subtensor_make_vector', 23) ('local_useless_elemwise', 15) ('local_mul_canonizer', 9) ('local_add_canonizer', 7) ...
        2 - 0.428s 45 (0.010s in global opts, 0.005s io_toposort) - 596 nodes - ('constant_folding', 15) ('local_useless_elemwise', 9) ('local_add_canonizer', 7) ('local_dimshuffle_lift', 6) ('local_flatten_lift', 3) ...
        3 - 0.094s 15 (0.006s in global opts, 0.005s io_toposort) - 563 nodes - ('constant_folding', 5) ('local_useless_elemwise', 3) ('local_dimshuffle_lift', 3) ('local_fill_sink', 2) ('MergeOptimizer', 1) ...
        4 - 0.081s 6 (0.001s in global opts, 0.004s io_toposort) - 551 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_fill_sink', 1) ('constant_folding', 1)
        5 - 0.078s 2 (0.000s in global opts, 0.004s io_toposort) - 547 nodes - ('MergeOptimizer', 1) ('local_fill_sink', 1)
        6 - 0.078s 2 (0.000s in global opts, 0.004s io_toposort) - 547 nodes - ('local_fill_to_alloc', 1) ('local_greedy_distributor', 1)
        7 - 0.077s 1 (0.000s in global opts, 0.004s io_toposort) - 545 nodes - ('MergeOptimizer', 1)
        8 - 0.076s 0 (0.000s in global opts, 0.004s io_toposort) - 545 nodes - 
       times - times applied - nb node created - name:
       0.698s - 51 - 0 - constant_folding
       0.191s - 5 - 32 - local_greedy_distributor
       0.130s - 40 - 48 - local_add_canonizer
       0.112s - 11 - 21 - local_mul_canonizer
       0.093s - 6 - 5 - MergeOptimizer
       0.089s - 98 - 229 - local_dimshuffle_lift
       0.035s - 18 - 54 - local_upcast_elemwise_constant_inputs
       0.027s - 17 - 34 - local_fill_sink
       0.021s - 45 - 0 - local_useless_elemwise
       0.016s - 12 - 0 - local_cut_gpu_host_gpu
       0.007s - 10 - 39 - local_shape_to_shape_i
       0.005s - 23 - 0 - local_subtensor_make_vector
       0.004s - 12 - 4 - local_fill_to_alloc
       0.004s - 8 - 16 - local_flatten_lift
       0.001s - 2 - 3 - local_neg_to_mul
       0.001s - 1 - 2 - local_subtensor_lift
       0.141s - in 61 optimization that where not used (display only those with a runtime > 0)
         0.024s - local_one_minus_erf2
         0.023s - local_one_minus_erf
         0.022s - local_mul_zero
         0.016s - local_func_inv
         0.010s - local_track_shape_i
         0.006s - local_fill_cut
         0.006s - local_remove_switch_const_cond
         0.006s - local_cast_cast
         0.005s - local_pow_canonicalize
         0.004s - local_mul_switch_sink
         0.004s - local_IncSubtensor_serialize
         0.003s - local_div_switch_sink
         0.002s - local_0_dot_x
         0.002s - local_useless_subtensor
         0.001s - local_useless_slice
         0.001s - local_lift_transpose_through_dot
         0.001s - local_incsubtensor_of_zeros
         0.001s - local_useless_inc_subtensor
         0.001s - local_dimshuffle_no_inplace_at_canonicalize
         0.001s - local_sum_div_dimshuffle
         0.000s - local_sum_prod_all_to_none
         0.000s - local_op_of_op
         0.000s - local_cut_useless_reduce
         0.000s - local_reduce_join
         0.000s - local_subtensor_merge
         0.000s - local_subtensor_of_alloc
         0.000s - local_subtensor_of_dot
         0.000s - local_useless_alloc
         0.000s - local_join_empty
         0.000s - local_join_make_vector
         0.000s - local_useless_inc_subtensor_alloc
         0.000s - local_setsubtensor_of_constants
         0.000s - local_join_1

   1.078973s - ('gpu_opt', 'SeqOptimizer', 12) - 0.012s
     SeqOptimizer      gpu_opt  time 1.079s for 548/475 nodes before/after optimization
       0.012s for fgraph.validate()
       0.278s for callback
       1.014383s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.010s
         EquilibriumOptimizer          gpu_local_optimizations
           time 1.014s for 12 passes
           nb nodes (start, end,  max) 552 779 779
           time io_toposort 0.067s
           time in local optimizers 0.755s
           time in global optimizers 0.091s
            0 - 0.402s 322 (0.001s in global opts, 0.004s io_toposort) - 552 nodes - ('local_gpu_elemwise_1', 118) ('constant_folding', 98) ('local_gpu_dimshuffle_0', 44) ('local_gpu_elemwise_0', 37) ('local_gpu_subtensor', 6) ...
            1 - 0.196s 100 (0.062s in global opts, 0.006s io_toposort) - 666 nodes - ('local_gpu_elemwise_1', 44) ('constant_folding', 23) ('local_gpu_dimshuffle_0', 12) ('local_gpu_elemwise_0', 8) ('local_gpu_gemm', 6) ...
            2 - 0.076s 26 (0.018s in global opts, 0.006s io_toposort) - 706 nodes - ('local_gpu_dimshuffle_0', 8) ('local_gpu_elemwise_0', 6) ('local_gpu_elemwise_1', 5) ('constant_folding', 4) ('local_gpu_careduce', 2) ...
            3 - 0.059s 21 (0.003s in global opts, 0.006s io_toposort) - 729 nodes - ('local_gpu_elemwise_0', 6) ('constant_folding', 6) ('local_gpu_dimshuffle_0', 3) ('local_gpu_flatten', 2) ('local_gpu_careduce', 2) ...
            4 - 0.046s 11 (0.004s in global opts, 0.006s io_toposort) - 750 nodes - ('local_gpu_flatten', 2) ('local_gpu_elemwise_1', 2) ('local_gpu_elemwise_0', 2) ('constant_folding', 2) ('local_gpu_dimshuffle_0', 1) ...
            5 - 0.039s 4 (0.003s in global opts, 0.006s io_toposort) - 758 nodes - ('local_gpu_careduce', 2) ('MergeOptimizer', 1) ('local_gpu_flatten', 1)
            6 - 0.032s 3 (0.000s in global opts, 0.006s io_toposort) - 760 nodes - ('local_gpu_flatten', 2) ('MergeOptimizer', 1)
            7 - 0.034s 1 (0.000s in global opts, 0.006s io_toposort) - 762 nodes - ('local_gpu_join', 1)
            8 - 0.032s 2 (0.000s in global opts, 0.005s io_toposort) - 773 nodes - ('MergeOptimizer', 1) ('local_gpu_elemwise_0', 1)
            9 - 0.035s 1 (0.000s in global opts, 0.005s io_toposort) - 775 nodes - ('local_gpu_careduce', 1)
           10 - 0.032s 1 (0.000s in global opts, 0.006s io_toposort) - 777 nodes - ('local_gpu_elemwise_0', 1)
           11 - 0.031s 0 (0.000s in global opts, 0.006s io_toposort) - 779 nodes - 
           times - times applied - nb node created - name:
           0.253s - 133 - 0 - constant_folding
           0.153s - 169 - 497 - local_gpu_elemwise_1
           0.091s - 8 - 0 - MergeOptimizer
           0.067s - 61 - 274 - local_gpu_elemwise_0
           0.049s - 68 - 149 - local_gpu_dimshuffle_0
           0.017s - 8 - 24 - local_gpu_careduce
           0.010s - 10 - 20 - local_gpu_flatten
           0.009s - 6 - 24 - local_gpu_dot22
           0.009s - 6 - 24 - local_gpu_gemm
           0.008s - 6 - 12 - local_gpu_subtensor
           0.007s - 4 - 12 - local_gpu_incsubtensor
           0.006s - 2 - 6 - local_gpu_lazy_ifelse
           0.005s - 2 - 4 - local_gpu_allocempty
           0.004s - 4 - 12 - local_gpualloc
           0.003s - 1 - 12 - local_gpu_join
           0.003s - 4 - 4 - local_gpualloc_memset_0
           0.152s - in 48 optimization that where not used (display only those with a runtime > 0)
             0.017s - local_track_shape_i
             0.016s - local_dnn_conv_alpha_merge
             0.015s - local_elemwise_alloc
             0.014s - local_dnn_convw_alpha_merge
             0.014s - local_dnn_convi_alpha_merge
             0.008s - local_dnn_conv_output_merge
             0.007s - local_dnn_convw_output_merge
             0.007s - local_dnn_convi_output_merge
             0.005s - local_gpu_dot_to_dot22
             0.005s - local_gpu_ger
             0.005s - local_gpu_gemv
             0.005s - local_gpu_conv
             0.005s - local_gpu_specifyShape_0
             0.004s - local_gpu_dot22scalar
             0.004s - local_gpu_eye
             0.004s - local_gpu_reshape
             0.004s - gpuScanOptimization
             0.004s - local_gpu_solve
             0.004s - local_gpu_advanced_incsubtensor1
             0.004s - local_gpu_advanced_subtensor1
             0.000s - local_gpu_contiguous_gpu_contiguous
             0.000s - local_gpu_elemwise_careduce
             0.000s - local_subtensor_make_vector
             0.000s - local_gpujoin_1

       0.063596s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.002s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.064s for 2 passes
           nb nodes (start, end,  max) 779 475 779
           time io_toposort 0.009s
           time in local optimizers 0.050s
           time in global optimizers 0.000s
            0 - 0.057s 158 (0.000s in global opts, 0.006s io_toposort) - 779 nodes - ('local_cut_gpu_host_gpu', 158)
            1 - 0.006s 0 (0.000s in global opts, 0.004s io_toposort) - 475 nodes - 
           times - times applied - nb node created - name:
           0.048s - 158 - 0 - local_cut_gpu_host_gpu
           0.002s - in 1 optimization that where not used (display only those with a runtime > 0)
             0.002s - constant_folding

       0.000969s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000010s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.586688s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.239s
   0.209742s - ('gpu_elemwise_fusion', 'FusionOptimizer', 15) - 0.001s
     FusionOptimizer
      nb_iter 2
      nb_replacement 51
      nb_inconsistency_replace 0
      validate_time 0.000631809234619
      callback_time 0.0160751342773
      time_toposort 0.00806093215942
   0.154740s - ('scan_eqopt2', 'EquilibriumOptimizer', 7) - 0.000s
     EquilibriumOptimizer      scan_eqopt2
       time 0.155s for 1 passes
       nb nodes (start, end,  max) 548 548 548
       time io_toposort 0.004s
       time in local optimizers 0.000s
       time in global optimizers 0.150s
        0 - 0.155s 0 (0.150s in global opts, 0.004s io_toposort) - 548 nodes - 
   0.121277s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 34) - 0.056s
   0.118755s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
     EquilibriumOptimizer      specialize
       time 0.119s for 3 passes
       nb nodes (start, end,  max) 557 548 558
       time io_toposort 0.012s
       time in local optimizers 0.079s
       time in global optimizers 0.013s
        0 - 0.041s 7 (0.004s in global opts, 0.004s io_toposort) - 557 nodes - ('local_shape_to_shape_i', 4) ('local_mul_specialize', 3)
        1 - 0.040s 6 (0.004s in global opts, 0.004s io_toposort) - 558 nodes - ('local_subtensor_make_vector', 4) ('constant_folding', 2)
        2 - 0.037s 0 (0.004s in global opts, 0.004s io_toposort) - 548 nodes - 
       times - times applied - nb node created - name:
       0.010s - 3 - 5 - local_mul_specialize
       0.005s - 2 - 0 - constant_folding
       0.001s - 4 - 5 - local_shape_to_shape_i
       0.001s - 4 - 0 - local_subtensor_make_vector
       0.074s - in 61 optimization that where not used (display only those with a runtime > 0)
         0.013s - crossentropy_to_crossentropy_with_softmax_with_bias
         0.010s - local_add_specialize
         0.009s - local_one_minus_erf2
         0.008s - local_one_minus_erf
         0.005s - local_elemwise_alloc
         0.005s - local_func_inv
         0.005s - local_div_to_inv
         0.004s - local_useless_elemwise
         0.003s - local_track_shape_i
         0.003s - local_abs_merge
         0.002s - local_elemwise_sub_zeros
         0.002s - local_cast_cast
         0.002s - local_alloc_unary
         0.001s - local_pow_specialize
         0.001s - local_mul_to_sqr
         0.001s - local_grad_log_erfc_neg
         0.000s - local_useless_subtensor
         0.000s - local_dimshuffle_lift
         0.000s - local_useless_slice
         0.000s - local_useless_inc_subtensor
         0.000s - local_sum_prod_mul_by_scalar
         0.000s - local_sum_div_dimshuffle
         0.000s - local_reduce_broadcastable
         0.000s - local_useless_alloc
         0.000s - local_opt_alloc
         0.000s - local_subtensor_merge
         0.000s - local_subtensor_of_dot
         0.000s - local_subtensor_of_alloc
         0.000s - local_neg_div_neg
         0.000s - local_join_empty
         0.000s - local_useless_inc_subtensor_alloc
         0.000s - local_join_make_vector
         0.000s - local_neg_neg
         0.000s - local_join_1

   0.080164s - ('stabilize', 'EquilibriumOptimizer', 6) - 0.000s
     EquilibriumOptimizer      stabilize
       time 0.080s for 2 passes
       nb nodes (start, end,  max) 545 548 548
       time io_toposort 0.008s
       time in local optimizers 0.060s
       time in global optimizers 0.008s
        0 - 0.041s 2 (0.004s in global opts, 0.004s io_toposort) - 545 nodes - ('Elemwise{log,no_inplace}(Elemwise{sub,no_inplace}(y subject to <function _is_1 at 0x7fb67a153b90>, sigmoid(x))) -> Elemwise{neg,no_inplace}(softplus(x))', 1) ('Elemwise{log,no_inplace}(sigmoid(x)) -> Elemwise{neg,no_inplace}(softplus(Elemwise{neg,no_inplace}(x)))', 1)
        1 - 0.039s 0 (0.004s in global opts, 0.004s io_toposort) - 548 nodes - 
       times - times applied - nb node created - name:
       0.001s - 1 - 2 - Elemwise{log,no_inplace}(Elemwise{sub,no_inplace}(y subject to <function _is_1 at 0x7fb67a153b90>, sigmoid(x))) -> Elemwise{neg,no_inplace}(softplus(x))
       0.001s - 1 - 3 - Elemwise{log,no_inplace}(sigmoid(x)) -> Elemwise{neg,no_inplace}(softplus(Elemwise{neg,no_inplace}(x)))
       0.066s - in 34 optimization that where not used (display only those with a runtime > 0)
         0.038s - local_greedy_distributor
         0.008s - crossentropy_to_crossentropy_with_softmax_with_bias
         0.006s - local_one_minus_erf2
         0.005s - local_one_minus_erf
         0.004s - local_sigm_times_exp
         0.002s - constant_folding
         0.002s - local_exp_over_1_plus_exp
         0.001s - local_grad_log_erfc_neg
         0.000s - local_0_dot_x
         0.000s - local_incsubtensor_of_zeros
         0.000s - local_useless_alloc
         0.000s - local_flatten_lift
         0.000s - local_subtensor_of_dot
         0.000s - local_useless_inc_subtensor_alloc
         0.000s - local_setsubtensor_of_constants
         0.000s - local_log1p
         0.000s - local_log_erfc
         0.000s - local_log_add

   0.072290s - ('gpu_after_fusion', 'SeqOptimizer', 17) - 0.000s
     SeqOptimizer      gpu_after_fusion  time 0.072s for 320/312 nodes before/after optimization
       0.000s for fgraph.validate()
       0.006s for callback
       0.059548s - ('gpu_local_optimizations', 'EquilibriumOptimizer', 2) - 0.000s
         EquilibriumOptimizer          gpu_local_optimizations
           time 0.059s for 4 passes
           nb nodes (start, end,  max) 324 328 330
           time io_toposort 0.011s
           time in local optimizers 0.036s
           time in global optimizers 0.002s
            0 - 0.020s 6 (0.000s in global opts, 0.003s io_toposort) - 324 nodes - ('local_gpu_elemwise_0', 2) ('constant_folding', 2) ('local_gpu_elemwise_1', 1) ('local_gpu_elemwise_careduce', 1)
            1 - 0.015s 2 (0.000s in global opts, 0.003s io_toposort) - 330 nodes - ('constant_folding', 2)
            2 - 0.013s 1 (0.002s in global opts, 0.003s io_toposort) - 328 nodes - ('MergeOptimizer', 1)
            3 - 0.011s 0 (0.000s in global opts, 0.003s io_toposort) - 328 nodes - 
           times - times applied - nb node created - name:
           0.009s - 2 - 10 - local_gpu_elemwise_0
           0.008s - 4 - 0 - constant_folding
           0.002s - 1 - 3 - local_gpu_elemwise_1
           0.002s - 1 - 0 - MergeOptimizer
           0.000s - 1 - 1 - local_gpu_elemwise_careduce
           0.017s - in 59 optimization that where not used (display only those with a runtime > 0)
             0.003s - local_elemwise_alloc
             0.002s - local_track_shape_i
             0.002s - local_dnn_conv_alpha_merge
             0.002s - local_dnn_convw_alpha_merge
             0.002s - local_dnn_convi_alpha_merge
             0.001s - local_dnn_conv_output_merge
             0.001s - local_dnn_convw_output_merge
             0.001s - local_dnn_convi_output_merge
             0.000s - local_gpu_lazy_ifelse
             0.000s - local_gpu_dimshuffle_0
             0.000s - local_gpu_ger
             0.000s - local_gpu_dot_to_dot22
             0.000s - local_gpu_gemv
             0.000s - local_gpu_conv
             0.000s - local_gpu_subtensor
             0.000s - local_gpu_specifyShape_0
             0.000s - local_gpu_dot22scalar
             0.000s - local_gpu_dot22
             0.000s - gpuScanOptimization
             0.000s - local_gpu_gemm
             0.000s - local_gpu_eye
             0.000s - local_gpu_reshape
             0.000s - local_gpu_solve
             0.000s - local_gpu_incsubtensor
             0.000s - local_gpu_flatten
             0.000s - local_gpu_advanced_incsubtensor1
             0.000s - local_gpu_advanced_subtensor1
             0.000s - local_gpu_allocempty
             0.000s - local_gpu_contiguous_gpu_contiguous
             0.000s - local_gpualloc_memset_0
             0.000s - local_gpu_careduce
             0.000s - local_subtensor_make_vector
             0.000s - local_gpujoin_1

       0.011812s - ('gpu_cut_transfers', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          gpu_cut_transfers
           time 0.012s for 2 passes
           nb nodes (start, end,  max) 328 312 328
           time io_toposort 0.005s
           time in local optimizers 0.004s
           time in global optimizers 0.000s
            0 - 0.007s 10 (0.000s in global opts, 0.003s io_toposort) - 328 nodes - ('local_cut_gpu_host_gpu', 10)
            1 - 0.004s 0 (0.000s in global opts, 0.003s io_toposort) - 312 nodes - 
           times - times applied - nb node created - name:
           0.003s - 10 - 0 - local_cut_gpu_host_gpu
           0.001s - in 1 optimization that where not used (display only those with a runtime > 0)
             0.001s - constant_folding

       0.000910s - ('InputToGpuOptimizer', 'InputToGpuOptimizer', 0) - 0.000s
       0.000006s - ('NoCuDNNRaise', 'NoCuDNNRaise', 1) - 0.000s

   0.058284s - ('elemwise_fusion', 'SeqOptimizer', 14) - 0.000s
     SeqOptimizer      elemwise_fusion  time 0.058s for 475/447 nodes before/after optimization
       0.000s for fgraph.validate()
       0.007s for callback
       0.054319s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
         FusionOptimizer
          nb_iter 2
          nb_replacement 17
          nb_inconsistency_replace 0
          validate_time 0.000212669372559
          callback_time 0.00696063041687
          time_toposort 0.00731205940247
       0.003956s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
         FusionOptimizer
          nb_iter 1
          nb_replacement 0
          nb_inconsistency_replace 0
          validate_time 0.0
          callback_time 0.0
          time_toposort 0.00351500511169

   0.045552s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
     SeqOptimizer      BlasOpt  time 0.046s for 548/557 nodes before/after optimization
       0.000s for fgraph.validate()
       0.003s for callback
       0.015112s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
         GemmOptimizer
          nb_iter 1
          nb_replacement 0
          nb_replacement_didn_t_remove 0
          nb_inconsistency_make 0
          nb_inconsistency_replace 0
          time_canonicalize 0.00584506988525
          time_factor_can 0
          time_factor_list 0
          time_toposort 0.00388407707214
          validate_time 0.0
          callback_time 0.0
       0.008277s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (548, 557, 2)
           init io_toposort 0.00406312942505
           loop time 0.00418090820312
           callback_time 0.00145292282104
       0.007314s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (548, 548, 6)
           init io_toposort 0.00399613380432
           loop time 0.00328397750854
           callback_time 0.00170397758484
       0.005572s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
         EquilibriumOptimizer          local_gemm_to_gemv
           time 0.006s for 1 passes
           nb nodes (start, end,  max) 557 557 557
           time io_toposort 0.004s
           time in local optimizers 0.000s
           time in global optimizers 0.000s
            0 - 0.006s 0 (0.000s in global opts, 0.004s io_toposort) - 557 nodes - 
       0.004767s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (557, 557, 0)
           init io_toposort 0.00394797325134
           loop time 0.000785112380981
           callback_time 0.0
       0.004492s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
         TopoOptimizer
           nb_node (start, end, changed) (557, 557, 0)
           init io_toposort 0.00396513938904
           loop time 0.000498056411743
           callback_time 0.0

   0.038466s - ('add_destroy_handler', 'AddDestroyHandler', 18) - 0.000s
   0.037867s - ('merge1', 'MergeOptimizer', 0) - 0.003s
     MergeOptimizer
       nb_fail 0
       replace_time 0.0250158309937
       validate_time 0.00283217430115
       callback_time 0.00911951065063
       nb_merged 267
       nb_constant 141
   0.034814s - ('ShapeOpt', 'ShapeOptimizer', 1) - 0.000s
   0.030796s - ('scan_eqopt1', 'EquilibriumOptimizer', 2) - 0.000s
     EquilibriumOptimizer      scan_eqopt1
       time 0.031s for 1 passes
       nb nodes (start, end,  max) 684 684 684
       time io_toposort 0.005s
       time in local optimizers 0.000s
       time in global optimizers 0.025s
        0 - 0.031s 0 (0.025s in global opts, 0.005s io_toposort) - 684 nodes - 
   0.024882s - ('cond_make_inplace', 'TopoOptimizer', 36) - 0.021s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 4)
       init io_toposort 0.00247287750244
       loop time 0.022369146347
       callback_time 0.0213062763214
   0.024163s - ('local_dnn_conv_inplace', 'TopoOptimizer', 26) - 0.013s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 8)
       init io_toposort 0.00244498252869
       loop time 0.0216240882874
       callback_time 0.0161254405975
   0.019071s - ('merge3', 'MergeOptimizer', 40) - 0.018s
     MergeOptimizer
       nb_fail 0
       replace_time 0.019061088562
       validate_time 0.0182957649231
       callback_time 0.0183427333832
       nb_merged 4
       nb_constant 4
   0.010521s - ('local_inplace_setsubtensor', 'TopoOptimizer', 21) - 0.005s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 4)
       init io_toposort 0.00242614746094
       loop time 0.00803899765015
       callback_time 0.00637745857239
   0.007457s - ('InplaceGpuBlasOpt', 'TopoOptimizer', 23) - 0.003s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 2)
       init io_toposort 0.00245189666748
       loop time 0.00491404533386
       callback_time 0.00375485420227
   0.006584s - ('scanOp_make_inplace', 'ScanInplaceOptimizer', 35) - 0.000s
   0.005869s - ('local_IncSubtensor_serialize', 'TopoOptimizer', 3) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (684, 684, 0)
       init io_toposort 0.00463795661926
       loop time 0.00120091438293
       callback_time 0.0
   0.005842s - ('gpu_scanOp_make_inplace', 'ScanInplaceOptimizer', 31) - 0.000s
   0.005783s - ('dimshuffle_as_view', 'TopoOptimizer', 19) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 4)
       init io_toposort 0.00270104408264
       loop time 0.00304198265076
       callback_time 0.00198650360107
   0.005762s - ('specialize_device', 'EquilibriumOptimizer', 13) - 0.000s
     EquilibriumOptimizer      specialize_device
       time 0.006s for 1 passes
       nb nodes (start, end,  max) 475 475 475
       time io_toposort 0.004s
       time in local optimizers 0.001s
       time in global optimizers 0.000s
        0 - 0.006s 0 (0.000s in global opts, 0.004s io_toposort) - 475 nodes - 
   0.005745s - ('gpua_scanOp_make_inplace', 'ScanInplaceOptimizer', 33) - 0.000s
   0.004700s - ('uncanonicalize', 'EquilibriumOptimizer', 11) - 0.000s
     EquilibriumOptimizer      uncanonicalize
       time 0.005s for 1 passes
       nb nodes (start, end,  max) 548 548 548
       time io_toposort 0.004s
       time in local optimizers 0.000s
       time in global optimizers 0.000s
        0 - 0.005s 0 (0.000s in global opts, 0.004s io_toposort) - 548 nodes - 
   0.004276s - ('crossentropy_to_crossentropy_with_softmax', 'FromFunctionOptimizer', 10) - 0.000s
   0.003217s - ('gpua_elemwise_fusion', 'FusionOptimizer', 29) - 0.000s
     FusionOptimizer
      nb_iter 1
      nb_replacement 0
      nb_inconsistency_replace 0
      validate_time 0.0
      callback_time 0.0
      time_toposort 0.00287103652954
   0.003215s - ('blas_opt_inplace', 'TopoOptimizer', 22) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00245404243469
       loop time 0.000666856765747
       callback_time 0.0
   0.003209s - ('gpuablas_opt_inplace', 'TopoOptimizer', 24) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00245809555054
       loop time 0.000656843185425
       callback_time 0.0
   0.003020s - ('random_make_inplace', 'TopoOptimizer', 38) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00275492668152
       loop time 0.000235080718994
       callback_time 0.0
   0.002978s - ('c_blas_destructive', 'TopoOptimizer', 25) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00242710113525
       loop time 0.000519990921021
       callback_time 0.0
   0.002808s - ('mrg_random_make_inplace', 'TopoOptimizer', 39) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00247311592102
       loop time 0.000306129455566
       callback_time 0.0
   0.002782s - ('make_ger_destructive', 'TopoOptimizer', 28) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00245904922485
       loop time 0.000293016433716
       callback_time 0.0
   0.002766s - ('local_inplace_incsubtensor1', 'TopoOptimizer', 20) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00248789787292
       loop time 0.000221967697144
       callback_time 0.0
   0.002742s - ('local_destructive', 'TopoOptimizer', 37) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00248312950134
       loop time 0.000226020812988
       callback_time 0.0
   0.002740s - ('local_gemm16_inplace', 'TopoOptimizer', 27) - 0.000s
     TopoOptimizer
       nb_node (start, end, changed) (312, 312, 0)
       init io_toposort 0.00249099731445
       loop time 0.000193119049072
       callback_time 0.0
   0.002570s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 32) - 0.000s
   0.000150s - ('merge2', 'MergeOptimizer', 16) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 0.000146150588989
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
   0.000055s - ('merge1.2', 'MergeOptimizer', 5) - 0.000s
     MergeOptimizer
       nb_fail 0
       replace_time 4.6968460083e-05
       validate_time 0.0
       callback_time 0.0
       nb_merged 0
       nb_constant 0
