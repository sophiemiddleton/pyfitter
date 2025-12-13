# ⚙️ `process.py` - Data Processing and Preparation

The `process.py` file is the data pipeline module. It defines the main class, `AnaProcessor`, which is responsible for **reading Mu2e EventNtuples**, applying event and track **selection cuts**, and preparing the final event data for the subsequent `zfit` minimization.

## 📝 `class AnaProcessor(Skeleton)`

This class inherits from the `pyutils` framework's `Skeleton` class, providing the foundational structure for parallel file processing.

### Initialization (`__init__`)

The initializer configures the data source, parallel processing settings, and specifies the exact Ntuple branches required for the analysis.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_list_path` | `str` | Path to the `.txt` file containing the list of Ntuple files to process. |
| `jobs` | `int` | Number of worker threads/processes to use for parallel processing (ideally 1 worker per file). |
| `cuts` | `list[bool]` | Boolean list determining which cuts defined in `CutManager` are active for this analysis run. |
| `location` | `str` | Specifies the file location (`'disk'` for remote/cvmfs, `'local'` otherwise). |
| `mom_lo`, `mom_hi` | `float` | Sets the lower and upper bounds for the momentum fit range, which are passed to the `Analyze` object. |

#### **Key Attributes Defined**

* `self.branches`: A dictionary specifying the exact Ntuple branches (e.g., `trk.nactive`, `trkqual.result`) that must be read from the input ROOT files. This optimizes I/O performance.
* `self.analyse`: An instance of the **`Analyze`** class, which handles the core, event-level selection logic.

### Core Method: `process_file(self, file_name)`

This method is the core worker function, automatically called by the parent `Processor` framework for each file.

1.  **Data Extraction:** A local `Processor` instance is created to handle the file I/O (using `uproot` and `awkward`).
2.  **Analysis Execution:** The raw data chunk is passed to `self.analyse.execute(data, file_name)`, which performs all track and event-level selection cuts.
3.  **Result Return:** Returns a dictionary (`results`) containing the filtered event data (`filtered_data`) and the file's cut flow statistics (`cut_stats`).

## 🛠️ Utility Functions

These functions handle post-processing tasks, primarily combining the output from multiple parallel workers before the data is passed to the fitter.

### `combine_cut_flows(cut_flow_list)`

* **Purpose:** Aggregates the cut flow statistics (events passing each cut) collected from every file processed by the parallel workers.
* **Logic:** Initializes a combined cut flow list using the first file's structure as a template, then iteratively sums the `events_passing` count for each defined cut across all worker results.
* **Output:** Recalculates the absolute and relative percentages, and prints the final combined cut flow to the terminal and saves it to a CSV file using `CutManager.print_cut_stats`.

### `combine_arrays(results)`

* **Purpose:** Takes the filtered `awkward.Array` of events from each file and concatenates them into a single, comprehensive array.
* **Input:** A list of dictionaries, where each dictionary contains the `"filtered_data"` from one input file.
* **Output:** A single `awkward.Array` containing all events that passed the cuts across all processed files. This array is the input for the fitting stage.

### `categorize_tracks(data, mismatch=False)`

* **Purpose:** Assigns a category ID to each track based on its **Monte Carlo truth information** (`startCode`, `genCode`). This is used to separate event types (e.g., signal, DIO, Cosmic) for categorized fitting.
* **Logic:** Uses vectorized `awkward` operations to efficiently match `startCode` and `genCode` to the definitions stored in `mom_components.py`.

### `count_particle_types(data)`

* **Purpose:** Calculates the raw event counts for different background and signal sources based on MC truth information.
* **Logic:** Uses specific `startCode` and geometric criteria (like rho position) to create boolean masks for each particle type (e.g., DIO, CE, RPC, Cosmic). Prints the raw yields for diagnostic purposes.
* **Output:** Returns a 1D `awkward.Array` where each element corresponds to the primary track's MC truth category, ready for use in the fitting module.

## 🚀 Main Execution (`main(args)` and `__main__`)

The main execution block defines user command-line arguments and orchestrates the entire analysis pipeline:

1.  **Argument Parsing:** Uses `argparse` to handle user-defined configurations (file list, jobs, fit type, ranges).
2.  **Processing:** Instantiates `AnaProcessor` and calls `ana_processor.execute()`, which performs the parallel file processing.
3.  **Pre-Fit Preparation:** Calls `combine_arrays` and `combine_cut_flows` to aggregate the results.
4.  **Data Selection:** Applies the final track selection (`TT_Front`) and prepares the necessary event properties (`mom_mag`, `time`) as `awkward` arrays.
5.  **Fitting:** Calls the appropriate fitting function (`Unbinned_fit_mom`, `Unbinned_fit_time`, or `Unbinned_2d_fit_mom_time`) from **`fit_module.py`** based on the user's `args.fittype`.
6.  **Interpretation:** If requested, calls the **`ResultsClass`** from **`results_module.py`** to write fitted data, calculate significance, and set upper limits.

