import pandas as pd
import torch
import xarray as xr
import numpy as np
import json
from torch.utils.data import Dataset

class ToyEra5Dataset(Dataset):
    def __init__(
        self,
        zarr_path,
        start_date,
        end_date,
        lead_time_set,
        input_variable_names,
        output_variable_names,
        stats_path,
        *,
        use_era5_stats: bool = True,
        levels=None,
    ):
        super().__init__()
        # Store dataset parameters
        self.input_vars = input_variable_names
        self.output_vars = output_variable_names
        self.lead_time_set = sorted(lead_time_set)  # Sort for consistency
        self.max_lead_time = max(self.lead_time_set)
        self.stats_path = stats_path
        self.use_era5_stats = use_era5_stats
        self.levels = levels

        # Load the zarr dataset
        self.ds = xr.open_zarr(zarr_path)
        
        # Convert dates to pandas datetime and select time range
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.ds = self.ds.sel(time=slice(self.start_date, self.end_date))
        
        # Get unavailable timestamps and create valid timestamp indices
        self.unavailable_timestamps = set(getattr(self.ds, 'unavailable_timestamps', []))
        self.valid_indices = self._get_valid_indices()
    
        # Load normalization statistics
        self.calculate_global_stats()
        
    def get_stats(self) -> dict:
        """Read the stats from the stats.json file

        Returns:
            dict: A dictionary containing mean and std for each variable and year.
        """

        if self.use_era5_stats:
            with open(self.stats_path) as stats_file:
                stats = json.load(stats_file)
            return stats

        with open(f"{self.zarr_path}all_variables_stats.json") as stats_file:
            stats = json.load(stats_file)

        return stats
    
    def calculate_global_stats(self):
        """
        Calculates the average mean and std for each variable over the given range of years.
        Returns:
            dict: A dictionary containing averaged mean and std for each variable.
        """
        self.global_stats = {}
        stats = self.get_stats()

        for variable, variable_stats in stats.items():
            # Use all years available in stats to compute global stats
            all_years = [int(year) for year in variable_stats["mean"] if year.isdigit()]
            start_year = min(all_years)
            end_year = max(all_years)

            # Check if the variable stats have level information
            has_levels = isinstance(variable_stats["mean"][str(start_year)], dict)
            
            if has_levels:
                self.global_stats[variable] = {"mean": {}, "std": {}}
                # Get levels from the first year's data
                stats_levels = list(variable_stats["mean"][str(start_year)].keys())
                dataset_levels = [str(level) for level in self.ds.level.values]
                
                # Find common levels between stats and dataset
                common_levels = set(stats_levels) & set(dataset_levels)
                for level in common_levels:
                    level_means = []
                    level_stds = []
                    
                    for year in range(start_year, end_year + 1):
                        level_means.append(variable_stats["mean"][str(year)][str(level)])
                        level_stds.append(variable_stats["std"][str(year)][str(level)])
                    
                    self.global_stats[variable]["mean"][str(level)] = sum(level_means) / len(level_means)
                    self.global_stats[variable]["std"][str(level)] = sum(level_stds) / len(level_stds)
            else:
                self.global_stats[variable] = {"mean": None, "std": None}
                yearly_means = []
                yearly_stds = []
                
                for year in range(start_year, end_year + 1):
                    yearly_means.append(variable_stats["mean"][str(year)])
                    yearly_stds.append(variable_stats["std"][str(year)])
                
                self.global_stats[variable]["mean"] = sum(yearly_means) / len(yearly_means)
                self.global_stats[variable]["std"] = sum(yearly_stds) / len(yearly_stds)

    def get_variable_names(self):
        """
        Returns a list of all variables in order, including level-specific variables.
        
        Returns:
            list: Ordered list of variable names, with level-specific variables formatted as 'variable_level'
        """
        variable_names = []
        
        # Process input variables
        for var in self.input_vars:
            data = self.ds[var].isel(time=0)  # Use first time step to check structure
            has_levels = 'level' in data.dims
            
            if isinstance(self.global_stats[var]["mean"], dict) and has_levels:
                # Multi-level variable
                if self.levels is not None:
                    for level in self.levels:
                        level_str = str(float(level))
                        if level_str in data.level.values.astype(str):
                            # Add level-specific variable name (convert Pa back to hPa for readability)
                            variable_names.append(f"{var}_{int(float(level))}")
            else:
                # Single-level variable
                variable_names.append(var)
        
        return variable_names

    def get_input_variable_names(self):
        """Returns the ordered list of input variable names."""
        return self.get_variable_names()

    def get_output_variable_names(self):
        """Returns the ordered list of output variable names."""
        # Store current input_vars
        temp_input_vars = self.input_vars
        
        # Temporarily set input_vars to output_vars
        self.input_vars = self.output_vars
        
        # Get variable names
        output_names = self.get_variable_names()
        
        # Restore input_vars
        self.input_vars = temp_input_vars
        
        return output_names
    
    def get_output_variable_map(self):
        """
        Returns a dictionary mapping output variable names to their indices.
        """
        return {var: i for i, var in enumerate(self.get_output_variable_names())}

    def _get_valid_indices(self):
        """
        Creates a list of valid time indices that can be used for training.
        A valid index is one where both the input time and all possible lead times are available.
        """
        valid_indices = []
        times = self.ds.time.values
        
        for i in range(len(times)):
            # Skip if current timestamp is unavailable
            if times[i] in self.unavailable_timestamps:
                continue
            
            # Check if all potential lead times are available
            is_valid = True
            for lead_time in self.lead_time_set:
                if i + lead_time >= len(times):
                    is_valid = False
                    break
                if times[i + lead_time] in self.unavailable_timestamps:
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
        
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        # Get the actual time index from valid_indices
        time_idx = self.valid_indices[idx]
        
        # Randomly select a lead time from the set
        lead_time = torch.randint(0, len(self.lead_time_set), (1,)).item()
        actual_lead_time = self.lead_time_set[lead_time]
        
        # Double check that both input and output times are valid
        output_idx = time_idx + actual_lead_time
        input_time = self.ds.time.values[time_idx]
        output_time = self.ds.time.values[output_idx]
        
        if input_time in self.unavailable_timestamps or output_time in self.unavailable_timestamps:
            raise ValueError(
                f"Invalid time pair selected: input_time={input_time}, output_time={output_time}. "
                f"This should not happen as indices were pre-filtered."
            )
        
        # Get input data
        input_data = []
        input_shapes = []
        for var in self.input_vars:
            data = self.ds[var].isel(time=time_idx)
            has_levels = 'level' in data.dims
            
            if isinstance(self.global_stats[var]["mean"], dict) and has_levels:
                # Handle multi-level variables
                if self.levels is not None:
                    # Process each specified level as a separate variable
                    for level in self.levels:
                        level_str = str(float(level))
                        if level_str not in data.level.values.astype(str):
                            print(f"Warning: Level {level/100} hPa ({level} Pa) not found in variable {var}.")
                            print(f"Available levels (Pa): {data.level.values}")
                            continue
                        level_data = data.sel(level=float(level))
                        mean = self.global_stats[var]["mean"][level_str]
                        std = self.global_stats[var]["std"][level_str]
                        normalized = (level_data - mean) / std
                        input_data.append(normalized.values)
                        input_shapes.append(normalized.values.shape)
                else:
                    raise ValueError("Levels must be specified for multi-level variables")
            else:
                # Handle single-level variables
                mean = self.global_stats[var]["mean"]
                std = self.global_stats[var]["std"]
                data = (data - mean) / std
                input_data.append(data.values)
                input_shapes.append(data.values.shape)
        
        # Get output data
        output_data = []
        output_shapes = []
        for var in self.output_vars:
            data = self.ds[var].isel(time=output_idx)
            has_levels = 'level' in data.dims
            
            if isinstance(self.global_stats[var]["mean"], dict) and has_levels:
                # Handle multi-level variables
                if self.levels is not None:
                    # Process each specified level as a separate variable
                    for level in self.levels:
                        level_str = str(float(level))
                        if level_str not in data.level.values.astype(str):
                            print(f"Warning: Level {level/100} hPa ({level} Pa) not found in variable {var}.")
                            print(f"Available levels (Pa): {data.level.values}")
                            continue
                        level_data = data.sel(level=float(level))
                        mean = self.global_stats[var]["mean"][level_str]
                        std = self.global_stats[var]["std"][level_str]
                        normalized = (level_data - mean) / std
                        output_data.append(normalized.values)
                        output_shapes.append(normalized.values.shape)
                else:
                    raise ValueError("Levels must be specified for multi-level variables")
            else:
                # Handle single-level variables
                mean = self.global_stats[var]["mean"]
                std = self.global_stats[var]["std"]
                data = (data - mean) / std
                output_data.append(data.values)
                output_shapes.append(data.values.shape)
        
        # Stack along the first dimension to create (num_variables, lat, lon)
        inputs = torch.tensor(np.stack(input_data), dtype=torch.float32)
        outputs = torch.tensor(np.stack(output_data), dtype=torch.float32)
        
        # Normalize lead time
        normalized_lead_time = torch.tensor(actual_lead_time / self.max_lead_time, 
                                          dtype=torch.float32)
        
        return {
            "input": inputs,
            "lead_time": normalized_lead_time,
            "output": outputs
        }
        

        

