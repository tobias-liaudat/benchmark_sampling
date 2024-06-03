


## Important

All elements in the parameter dicitonary should be lists!


## Dataset

To call the dataset with different arguments, we can
```bash
benchopt run -d "natural_images[inv_problem=[test]]"
```

There we will overwrite the `inv_problem` variable with `test`.

If we want to run it on two inverse problems, we would need to run:
```bash
benchopt run -d "natural_images[inv_problem=[test, test2]]"
```


## Runnning benchmarks

We need to set the number of iterations (`-n`) and the timeout (`--timeout`) with the parameters:
```bash
run benchmark_sampling/ -d natural_images -s pnp-ula  -n 10 --timeout 10000
```

