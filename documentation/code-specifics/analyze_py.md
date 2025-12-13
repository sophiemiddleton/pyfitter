# 🔎 `analyze.py` - Core Selection Logic

The `analyze.py` module defines the `Analyze` class, which manages the application of all physics-driven event and track selection cuts. It acts as the central hub for defining cut masks using utility functions from `pyutils.pyselect` and managing the cut flow with `cut_manager.py`.

## 📝 `class Analyze`

The primary class that handles the core analysis workflow for a single data chunk (one file or a subset of events).

### Initialization (`__init__`)

Sets up logging and initializes key utility classes.

* `self.logger`: An instance of `pylogger.Logger` for controlled output.
* `self.selector`: An instance of `pyselect.Select`, providing low-level, vectorized functions to generate boolean masks for cuts.
* `self.sign`: The charge sign of the particle of interest (e.g., `"minus"` for electrons, `"plus"` for positrons), which dictates the initial particle ID cut.
* `self.switch`: A list of booleans passed from `process.py` that globally enables or disables individual cuts.

### Core Method: `define_cuts(self, data, cut_manager)`

This function defines the boolean masks for all individual cuts and registers them with the provided `CutManager` instance.

> **Note on Mu2e Tracks:** Many cuts are defined at the **track segment** level (`trksegs`) or the **track fit parameters** (`trksegpars_lh`), but must be aggregated to form a single **track-level** mask before being applied to the event data. The general pattern is:
>
> `cut_mask = ak.all(~at_trk_front | cut_mask_trksegs, axis=-1)`
>
> This ensures that all track segments that intersect the tracker entrance (`at_trk_front`) must pass the cut, making the final mask an event-level veto if any track segment fails.

| Cut Index (Switch) | Cut Name | Description | Data Field Used |
| :--- | :--- | :--- | :--- |
| `[0]` | `is_reco_electron`/`is_reco_positron` | Selects tracks based on the reconstructed charge sign (`trk.pdg`). | `data["trk"]` |
| `[1]` | `has_downstream` | Ensures tracks are moving in the positive Z direction (`p_z > 0`) through the tracker. | `data['trkfit']` |
| `[2]` | `good_trkqual` | Ensures the track fit quality is above a threshold (e.g., $> 0.2$). | `data["trk"]` |
| `[3]` | `good_trkpid` | New Track PID cut (e.g., $> 0.8$). | `data["trk"]` |
| `[4]` | `has_hits` | Requires a minimum number of active hits in the tracker (e.g., $\geq 20$). | `data["trk"]` |
| `[5]` | `within_t0` | Restricts the track time at the tracker mid-plane (e.g., $500 < t_0 < 1650$ ns). | `data['trkfit']["trksegs"]` |
| `[6]` | `within_t0err` | Requires the track time error from the loop helix fit to be small (e.g., $< 0.9$ ns). | `data['trkfit']["trksegpars_lh"]` |
| `[7]` | `within_lhr_max` | Limits the maximum radius of the track helix (e.g., $450 < R_{\text{max}} < 680$ mm). | `data['trkfit']["trksegpars_lh"]` |
| `[8]` | `within_d0` | Limits the distance of closest approach to the beamline (e.g., $d_0 < 100$ mm). | `data['trkfit']["trksegpars_lh"]` |
| `[9]` | `within_pitch_angle` | Limits the track's pitch angle ($\tan(\theta_{\text{Dip}})$) to a specific range. | `data['trkfit']["trksegpars_lh"]` |
| `[10]` | `has_st` | Veto based on the number of straw tubes (Nst) hit. | `data['trkfit']` |
| `[11]` | `no_opa` | Veto for tracks associated with the Outer Proton Absorber (OPA). | `data['trkfit']` |
| `[12]` | `within_t0_early` | An early time cut used for specific background studies (e.g., $0 < t_0 < 700$ ns). | `data['trkfit']["trksegs"]` |
| `[13]` | `no_crv_veto` | Veto events if the track time is close to a Cosmic Ray Veto (CRV) coincidence ($\Delta t < 150$ ns). | `data['trkfit']["trksegs"]`, `data["crv"]` |

### Core Method: `apply_cuts(self, data, cut_manager, ...)`

This method executes the selection process:

1.  **Mask Combination:** Calls `cut_manager.combine_cuts(active_only=True)` to produce a single boolean mask (`trk_mask`) representing the logical AND of all active cuts.
2.  **Track Selection:** Applies the `trk_mask` to the relevant track-level branches (`'trk'`, `'trkfit'`, `'trkmc'`) of the input `awkward.Array`, discarding tracks that failed the criteria.
    * **Crucial Step**  The selection process reduces the number of tracks per event (`events × tracks × properties`).
3.  **Event Cleanup:** Applies an event-level filter: `data_cut = data_cut[ak.any(trk_mask, axis=-1)]`. This removes entire events that contain zero remaining tracks after the track-level cuts are applied.

### Orchestration: `execute(self, data, file_id, ...)`

This method ties the entire process together for a single file:

1.  Initializes a file-specific `CutManager`.
2.  Calls `self.define_cuts` to establish all masks.
3.  Calls `cut_manager.calculate_cut_stats` to track the event count after each progressive cut.
4.  Calls `self.apply_cuts` to physically filter the data.
5.  Prints the final cut flow statistics and returns the results dictionary.
