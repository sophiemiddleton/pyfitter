# Weighted Fits and Limit Setting 

## Weighted Fit Builder

```
python weighted_fit_builder.py --fit-type 2d --components dio=file_lists/DIOtail95_MDC2025an_best_nomix.txt cosmic=file_lists/Cosimcs_MDC2025an_nomix.txt rpc=file_lists/ExtRPC_MDC2025an_nomix.txt   --variable recomom_ttfront --fit-range-lo 97  --fit-range-hi 110 --time-range-lo 475 --time-range-hi 1650 --jobs 16 --export-npz Run-1A-bkg.npz
```

## Simple Limit Setting


```
python sensitivity_scan/simple_limit.py 
```

> [NOTE:] ensure that the .npz file this file is looking for is the same as that outputed from the previous stage