


## Important

All elements in the parameter dicitonary should be lists!



## Dependencies

Extra

```bash
pip install arviz
pip install statsmodels
pip install pyiqa
```

`pyiqa` is required to compute the LPIPS metric.

If arviz has some problems with scipy, try downgrading scipy like this:
```bash
pip install scipy==1.12
```

If there's problems with pyiqa / an error like "ImportError: cannot import name 'packaging' from 'pkg_resources' try downgrading setuptools to version
```bash
pip install setuptools==69.5.1
```

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



## Code snippets

This snippet is helpful if you want to use the awesome save_final_results feature of benchopt (`https://github.com/benchopt/benchopt/pull/722`) to save images and then work on the parquet file, I wrote this nice little function to help you get figures instantly.

```python
def plot_grid_results(benchmark_file, max_cols=10, figsize=(10,10)):
    """Plot a grid of image results from a benchmark."""
    df = pd.read_parquet(benchmark_file)
    
    # Create a DataFrame from the Series with parsed parameters
    df = pd.concat([pd.DataFrame(list(df['solver_name'].apply(parse_name))), df], axis=1)
    df = df.convert_dtypes()
    # FIXME: https://github.com/benchopt/benchopt/issues/734
    df = df.drop("version-numpy", axis=1)
    
    fixed_columns = df.columns[df.apply(pd.Series.nunique) == 1]
    fixed_params = {c: df.loc[0, c] for c in fixed_columns}
    
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    var_params_names = df.columns[df.apply(pd.Series.nunique) != 1]
    
    fig = plt.figure(figsize=figsize)
    max_cols = 10
    n_img = len(df["solver_name"].unique())
    ncols = min(n_img, max_col)
    nrows = (n_img // max_cols) + 1
    
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows,ncols), axes_pad=0.1)
    
    for ax, solver_name in zip(grid, df["solver_name"].unique()):
        
        sub_df = df[df["solver_name"] == solver_name]
        args = parse_name(solver_name)
        filter_name = {k:v for k, v in args.items() if k in var_params_names}
        result_file = "../"+list(sub_df["final_results"])[0]
        img = np.load(Path(result_file).resolve(), allow_pickle=True)
        img = img.squeeze(0).squeeze(0).abs().numpy()
        ax.imshow(img, cmap="gray", origin='lower')
        ax.axis('off')
        sub_df
        ax.set_title(" ".join([f"{k}={v}" for k,v in filter_name.items()]))
    return fixed_params
```