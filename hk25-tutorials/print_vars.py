from IPython.display import display, Markdown
import pandas as pd
import xarray as xr
import sys
import argparse

def print_var_attributes(dir_file):
    """
    Print attributes for all variables in an xarray dataset in a clean format as a text file.
    
    Args:
    -----------
    dir_file (str):
        Path to zarr dataset containing variables to examine
    
    Returns:
    -----------
    txt (str):
        Text file of var details.
    
    Example command:
    -----------
    $ python print_vars.py 'path/to/zarr/file' --output 'path/to/my/output_name.txt'
    """
    ds = xr.open_zarr(dir_file)
    
    with open(args.output, 'w') as f:
        sys.stdout = f
    
        # Dataset overview
        print(f"Dataset dimensions: {dict(ds.sizes)}")
        print(f"Number of variables: {len(ds.data_vars)}")
        # Print vertical levels if they exist
        if 'lev' in ds:
            print(f"Vertical levels: {ds['lev'].values}")
        print("-" * 80)
        
        # Process each variable
        for var_name in ds.data_vars:
            var = ds[var_name]
            
            # Display variable information with bold name
            display(Markdown(f"\n## **{var_name}**"))
            print(f"Dimensions: {dict(var.sizes)}")
            print(f"Data type: {var.dtype}")
            
            # Display attributes in a table if they exist
            if var.attrs:
                print("\nAttributes:")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 200)  # or float('inf') with recent versions
                pd.set_option('display.max_colwidth', None)
                attrs_df = pd.DataFrame(list(var.attrs.items()), 
                                    columns=["Name", "Value"])
                display(attrs_df)
            else:
                print("\nNo attributes found for this variable")
            
            print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print attributes for all variables in an xarray dataset in a clean format as a text file.")
    parser.add_argument("dir_file", help="Path to the Zarr file")
    parser.add_argument("--output", default="output.txt", help="Save output as")
    
    args=parser.parse_args()
    print_var_attributes(args.dir_file)