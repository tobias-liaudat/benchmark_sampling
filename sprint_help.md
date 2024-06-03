


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

