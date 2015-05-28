#section support_code


#section support_code_apply

int APPLY_SPECIFIC(gated_unit_main)(CudaNdarray* inp,
                                    CudaNdarray* inp_to_hidden,
                                    CudaNdarray** output)
{
  // XXX this is horribly unsafe.
  // There NEEDS to be checking for all allocation steps
  printf("Hello world! from a cop");
  if (*output) Py_DECREF(*output);
  npy_intp dims[2];
  dims[0] = 6; dims[1] = 3;
  *output = (CudaNdarray *)CudaNdarray_New();
  //if ((NULL == *output
  CudaNdarray_alloc_contiguous(*output, 2, dims);

  CudaNdarray_gemm(1.0f, inp, inp_to_hidden, 0.0f, *output);

  return 0;
}
