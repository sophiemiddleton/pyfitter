# The `py-fitter` Framework

# The Code:

The code is currently object orientated with a set of distinct classes:

* `process.py` and `analyze.py`- the driver functions. The user can define several input parameters, check the default settings (at the bottom of the Main.py script).
* `cut_module.py` - takes in an opt to a list of cuts, applies cuts
* `fit_module.py` - runs unbinned ML fits to input data
* `recoplot_module.py` - plots reconstructed information
* `_PDF_module.py` - sets of PDFs to be input into the fit module ( mom, time)
* `_component.py` - user input list of chosen PDF names, parameters and draw options (mom, time)
* `results_module.py`- runs functions to interpret the result (e.g. significance tests and limit setting)

The latest version of the code >= v2_00_00 requires Mu2e's pyutils be within the users working directory.

# Running:

To run for example:

```
python process.py --file /exp/mu2e/app/users/sophie/analysis/LikelihoodAnalysis/py-fitter/filelist.txt --cat 1 --mismatch 1 --interpret 1
```

With a file list stored in the mentioned .txt file.

* The process function imports the given root NTuple (s) via the use of Mu2e's pyutils (maintained by the Mu2e Analysis tools group). This therefore assumes input is an up-to-date pyuyils. To get pyuyils in your python env:

```
pip install git+https://github.com/Mu2e/pyutils.git 

```

this is already installed in the standard python env mentioned above.


##  🚀 Code Documentation: `process.py`

`process.py` is our driving function it is here that we configure the data imports, likelihood assumptions and output the results of the analysis.

### 📦 Imports

The code relies heavily on several scientific computing and data analysis libraries, as well as several custom modules.

| Module | Description |
| :--- | :--- |
| `hist` | A library for handling histograms (though not explicitly used for filling here, it's often used for result presentation). |
| `gc`, `sys`, `datetime` | Standard Python utilities for memory management, system interactions, and timing. |
| `numpy` (`np`) | Fundamental package for numerical computation in Python. |
| `matplotlib.pyplot` (`plt`) | Used for plotting and visualization (though plotting functions are likely within the custom modules). |
| `uproot` | Used to read and process ROOT files (common format in particle physics) directly into Python structures. |
| `awkward` (`ak`) | Used to handle array data with nested, variable-length structures, especially from `uproot`. |
| `argparse` | Standard library for parsing command-line arguments. |
| `csv` | Standard library for working with CSV files. |
| `fit_module` | **Custom:** Contains the core unbinned fitting functions (e.g., `Unbinned_fit_mom`, `Unbinned_fit_time`). |
| `results_module` | **Custom:** Defines `ResultsClass` for interpreting, saving, and analyzing fit results (e.g., calculating significance, setting limits). |
| `analyze` | **Custom:** Defines the `Analyze` class, which handles the analysis logic and selection cuts applied to the extracted data. |
| `mom_components` | **Custom:** Likely a dictionary or module defining the components (signal, background) used in the momentum fits. |
| `pyutils.*` | **Custom Framework:** Provides utility classes for data processing, plotting, printing, selection, and vector operations. |

---

### ⚙️ Class: `AnaProcessor(Skeleton)`

This class is the core component for data extraction and file-level processing. It inherits from `Skeleton` (part of the `pyutils.pyprocess` framework) to fit into a multi-processing/multi-threading structure provided by `pyutils.pyprocess.Processor`.

#### `__init__(self, file_list_path, jobs=1, cuts=[], location='disk')`

* **Purpose:** Initializes the processor, sets up file-handling parameters, and defines the branches to be extracted from the input ntuple files.
* **Parameters:**
    * `file_list_path` (str): Path to the file containing a list of data files to process.
    * `jobs` (int, default=1): The maximum number of worker processes/threads to use.
    * `cuts` (list, default=[]): A list of switches (likely booleans) to control the analysis cuts in the `Analyze` module.
    * `location` (str, default='disk'): Specifies the file location (e.g., 'disk' or 'remote').
* **Key Attributes:**
    * `self.branches`: A dictionary mapping top-level ntuple branches (e.g., `"evt"`, `"trk"`) to a list of sub-branches/leaves to be read.
        > **Note:** The branches listed (`trk.nactive`, `trk.pdg`, `crvcoincs.time`, etc.) are characteristic of particle physics experiments (e.g., Muon $g-2$, Mu2e, or similar) where detailed event information is stored. 
    * `self.tree_path`: The name of the ROOT tree/TTree containing the data (`"ntuple"`).
    * `self.max_workers`: Sets the concurrency limit for the `pyprocess.Processor`.
    * `self.analyse`: An instance of the custom `Analyze` class, initialized with the analysis cuts.

#### `process_file(self, file_name)`

* **Purpose:** The method called by the `Processor` framework to handle a single input file. It reads the data, passes it to the analysis logic, and returns the result.
* **Process:**
    1.  Creates a temporary `pyutils.pyprocess.Processor` instance to handle data extraction for the specific file.
    2.  Calls `processor.process_data()` to read the specified `self.branches` from the `file_name` into an Awkward Array structure (`data`).
    3.  Passes the extracted `data` to `self.analyse.execute()`, which applies cuts and performs file-level data manipulation.
    4.  Runs `gc.collect()` to explicitly free up memory.
    5.  Returns the `results` (likely a subset of filtered Awkward Arrays or histograms).

---

### 📊 Utility Functions

#### `combine_arrays(results)`

* **Purpose:** Takes the list of results (Awkward Arrays) returned by `AnaProcessor.process_file` for all processed files and concatenates them into a single, large Awkward Array.
* **Mechanism:** Uses `ak.concatenate()` for efficient merging of nested, ragged array data.

#### `categorize_tracks(data, mismatch=False)`

* **Purpose:** Applies a set of classification rules based on Monte Carlo (MC) truth information (`trkmc` branches) to assign a category (index) to each track. This is crucial for separating signal and background in the subsequent fits.
* **Mechanism:**
    1.  Filters track information based on `rank == 0` (e.g., the primary or best-matched MC track) and `nhits > 0`.
    2.  Extracts key MC classification variables: `startCode` and `genCode`.
    3.  Iterates over categories defined in the `mom_components` module.
    4.  Assigns an integer category (starting from 1) to each track that matches the defined `startCode` and `genCode` combinations for that category.
    5.  Returns an Awkward Array of category indices.

---

### 🏃 Main Execution Flow

#### `main(args)`

This function orchestrates the entire analysis, from processing files to running the final fit.

1.  **File Processing:**
    * An `AnaProcessor` instance is created with user-specified file list, jobs, and cuts (`old` list is hardcoded, but `new` is also available).
    * `ana_processor.execute()` runs the multi-file processing, returning a list of results (filtered arrays).
    * `pre_fit = combine_arrays(results)` merges all results into a single Awkward Array.

2.  **Data Selection and Preprocessing:**
    * A `Select` utility is used to find tracks that intersect a specific detector plane/surface (`TT_Front`).
    * The track data (`trkfit_ent`) is masked to include only those passing the `trk_front` selection.
    * If categorization is requested (`args.cat == 1`):
        * `categorize_tracks` is called.
        * The track categories are also masked by the `trk_front` selection.
    * The magnitude of the track momentum (`mom_mag`) and time (`time`) are calculated using the `Vector` utility and cleaned (dropping `NaN` or `None` values).

3.  **Unbinned Maximum Likelihood Fitting:**
    * The type of fit is determined by `args.fittype`.
    * **`mom1D` (1D Momentum Fit):** Calls `Unbinned_fit_mom` (from `fit_module`) using `mom_mag` and optional `track_cat`.
        * If `args.interpret == 1`, `ResultsClass` is used to save fitted data, calculate statistical significance, and optionally set an upper limit (`args.setlimit == 1`) using a method like **Asymptotic Limits (`asym`)**. 
    * **`time1D` (1D Time Fit):** Calls `Unbinned_fit_time` using `time` and optional `track_cat`.
    * **`momtime2D` (2D Momentum/Time Fit):** Placeholder/FIXME, currently raises an exception if selected.

#### `if __name__ == "__main__":`

* **Command Line Interface (CLI):** Sets up `argparse` to define and parse command-line arguments, allowing users to configure the analysis run.
* **Key Arguments:**
    * `--file`: Input file list.
    * `--jobs`: Number of processes.
    * `--fittype`: Analysis type (`mom1D`, `time1D`, `momtime2D`).
    * `--fitrange_low`/`--fitrange_hi`: The range $[P_{min}, T_{min}]$ to fit in momentum and time.
    * `--interpret`: Flag to enable significance/limit calculation.
    * `--cat`: Flag to enable Monte Carlo-based track categorization.
* Calls `PrintArgs` to show the user's configuration and then executes `main(args)`.


##  🚀 Code Documentation: `analyze.py`
