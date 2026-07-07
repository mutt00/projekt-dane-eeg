import mne
from mne.preprocessing import ICA, annotate_amplitude
from mne_icalabel import label_components
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


#mne.viz.set_browser_backend("qt")
matplotlib.use("Agg")

mne.set_log_level("CRITICAL")


PROJECT_PATH = Path(__file__).parent
DATA_DIR = "data"
DERIV_DIR = "derivatives"

RESAMPLE_FREQ = 250.0
L_FREQ, H_FREQ = 0.1, 30.0
NOTCH_FREQ = 60.0

MASTOIDS = ["M1", "M2"]
ORIGINAL_REFS = ["Fz"]
MONTAGE = "easycap-M1"

REWP_CHAN = "FCz"
REWP_TMIN, REWP_TMAX = 0.240, 0.340
EPOCH_TMIN, EPOCH_TMAX = -0.200, 0.600
BASELINE = (-0.200, 0.0)

PTP_THRESH = 150e-6
GRAD_THRESH = 40e-6
CHAN_REJECT_PROP = 0.10
MAX_BAD_CHANS = 3

ICA_L_FREQ = 1.0
ICLABEL_PROB = 0.80

EVENT_ID = {
    "unlearnable/win":   7,
    "unlearnable/loss":  8,
    "learnable/win":    17,
    "learnable/loss":   18,
}

ANNOT_TO_ID = {
    "Stimulus/S  7":  7,
    "Stimulus/S  8":  8,
    "Stimulus/S 17": 17,
    "Stimulus/S 18": 18,
}


def mean_amp_uv(evk):
    return (evk.copy().pick(REWP_CHAN)
            .crop(REWP_TMIN, REWP_TMAX).data.mean() * 1e6)


def process_subject(subject: str) -> dict | None:
    print(f"\n===== {subject}")

    vhdr = (PROJECT_PATH / DATA_DIR / subject / "eeg" /
            f"{subject}_task-differentdoors_eeg.vhdr")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)

    raw.resample(sfreq=RESAMPLE_FREQ, npad="auto")

    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="fir",
               phase="zero", fir_design="firwin", verbose=False)
    
    raw.add_reference_channels(ORIGINAL_REFS)
    raw.set_eeg_reference(ref_channels=MASTOIDS, verbose=False)
    raw.drop_channels(MASTOIDS)
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")

    automatic_rank = mne.compute_rank(raw, rank="info")["eeg"]
    real_rank = automatic_rank - 1
    print(f"Rank for ICA: {real_rank}")

    raw_ica = raw.copy().filter(l_freq=ICA_L_FREQ, h_freq=None, verbose=False)

    annot_bad, _ = annotate_amplitude(raw_ica, peak=1000e-6, bad_percent=50)
    raw_ica.set_annotations(raw_ica.annotations + annot_bad)

    ica = ICA(n_components=real_rank, method="infomax",
              fit_params=dict(extended=True),
              random_state=97, max_iter="auto")
    ica.fit(raw_ica, reject_by_annotation=True)

    labels = label_components(raw_ica, ica, method="iclabel")
    exclude_idx = [
        i for i, (lab, prob) in enumerate(
            zip(labels["labels"], labels["y_pred_proba"]))
        if lab in ("eye blink", "muscle artifact") and prob > ICLABEL_PROB
    ]
    ica.exclude = exclude_idx
    ica.apply(raw)
    print(f"ICA components removed: {len(exclude_idx)} ({exclude_idx})")

    events, _ = mne.events_from_annotations(raw, event_id=ANNOT_TO_ID,
                                            verbose=False)
    epochs = mne.Epochs(raw, events, event_id=EVENT_ID,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=BASELINE,
                        preload=True, reject=None, verbose=False)

    data = epochs.get_data()
    ptp = data.max(axis=2) - data.min(axis=2)
    grad = np.abs(np.diff(data, axis=2)).max(axis=2)
    bad_mask = (ptp > PTP_THRESH) | (grad > GRAD_THRESH)

    chan_bad_prop = bad_mask.mean(axis=0)
    noisy_chans = [epochs.ch_names[i]
                   for i, p in enumerate(chan_bad_prop)
                   if p > CHAN_REJECT_PROP]

    print(f"Median PTP: {np.median(ptp)*1e6:.1f} µV")
    print(f"Median grad: {np.median(grad)*1e6:.2f} µV/sample")
    print(f"Fraction failing PTP: {(ptp > PTP_THRESH).mean():.3f}")
    print(f"Fraction failing grad: {(grad > GRAD_THRESH).mean():.3f}")
    for i, ch in enumerate(epochs.ch_names):
        if chan_bad_prop[i] > CHAN_REJECT_PROP:
            print(f"  {ch}: {chan_bad_prop[i]*100:.1f}% bad")
        
    if len(noisy_chans) > MAX_BAD_CHANS:
        print(f"{subject} excluded: {len(noisy_chans)} noisy channels")
        return None

    good_ch_idx = [i for i, ch in enumerate(epochs.ch_names)
                   if ch not in noisy_chans]
    drop_epoch = bad_mask[:, good_ch_idx].any(axis=1)
    epochs.drop(np.where(drop_epoch)[0], reason="artifact", verbose=False)

    epochs.info["bads"] = noisy_chans
    epochs.interpolate_bads(reset_bads=True, verbose=False)

    print(f"Noisy channels interpolated: {noisy_chans}")
    print(f"Epochs dropped: {int(drop_epoch.sum())} / {len(drop_epoch)}")

    evokeds = {}
    for cond in ("learnable", "unlearnable"):
        win = epochs[f"{cond}/win"].average()
        loss = epochs[f"{cond}/loss"].average()
        diff = mne.combine_evoked([win, loss], weights=[1, -1])
        evokeds[cond] = dict(win=win, loss=loss, diff=diff)

    rewp_learn = mean_amp_uv(evokeds["learnable"]["diff"])
    rewp_unlearn = mean_amp_uv(evokeds["unlearnable"]["diff"])
    delta_rewp = rewp_learn - rewp_unlearn

    print(f"{subject} RewP (learnable):   {rewp_learn:+.2f} µV")
    print(f"{subject} RewP (unlearnable): {rewp_unlearn:+.2f} µV")
    print(f"{subject} ΔRewP:              {delta_rewp:+.2f} µV")

    subj_deriv = PROJECT_PATH / DERIV_DIR / subject
    subj_deriv.mkdir(parents=True, exist_ok=True)
    for cond, d in evokeds.items():
        for kind, evk in d.items():
            evk.save(subj_deriv / f"{subject}_{cond}_{kind}_ave.fif",
                     overwrite=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for j, cond in enumerate(("unlearnable", "learnable")):
        mne.viz.plot_compare_evokeds(
            {"win": evokeds[cond]["win"],
             "loss": evokeds[cond]["loss"],
             "difference": evokeds[cond]["diff"]},
            picks=REWP_CHAN, axes=axes[0, j], show=False,
            title=f"{cond.capitalize()} ({subject})")
        evokeds[cond]["diff"].plot_topomap(
            times=[(REWP_TMIN + REWP_TMAX) / 2],
            average=REWP_TMAX - REWP_TMIN,
            axes=axes[1, j], show=False, colorbar=False)
    #fig.savefig(subj_deriv / f"{subject}_rewp_diagnostic.png", dpi=200)
    #plt.close(fig)

    return {
        "subject": subject,
        "rewp_learnable": rewp_learn,
        "rewp_unlearnable": rewp_unlearn,
        "delta_rewp": delta_rewp,
        "n_ica_excluded": len(exclude_idx),
        "n_bad_chans": len(noisy_chans),
        "n_epochs_dropped": int(drop_epoch.sum()),
        "n_epochs_kept": len(epochs),
    }, evokeds


def main():
    subjects = sorted(p.name for p in (PROJECT_PATH / DATA_DIR).glob("sub-*")
                      if p.is_dir())
    print(f"Found {len(subjects)} subjects")

    all_measures = []
    all_evokeds = {
        "learnable":   {"win": [], "loss": [], "diff": []},
        "unlearnable": {"win": [], "loss": [], "diff": []},
    }

    for sub in subjects:
        try:
            out = process_subject(sub)
        except Exception as e:
            print(f"{sub} failed: {type(e).__name__}: {e}")
            continue
        if out is None:
            continue

        measures, evokeds = out
        all_measures.append(measures)
        for cond in ("learnable", "unlearnable"):
            for kind in ("win", "loss", "diff"):
                all_evokeds[cond][kind].append(evokeds[cond][kind])

    deriv_root = PROJECT_PATH / DERIV_DIR
    deriv_root.mkdir(exist_ok=True)
    pd.DataFrame(all_measures).to_csv(
        deriv_root / "group_measures.tsv", sep="\t", index=False)

    grand = {cond: {k: mne.grand_average(v) for k, v in d.items()}
             for cond, d in all_evokeds.items()}
    for cond, d in grand.items():
        for kind, evk in d.items():
            evk.save(deriv_root / f"grand_{cond}_{kind}_ave.fif",
                     overwrite=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for j, cond in enumerate(("unlearnable", "learnable")):
        mne.viz.plot_compare_evokeds(
            {"win": grand[cond]["win"],
             "loss": grand[cond]["loss"],
             "difference": grand[cond]["diff"]},
            picks=REWP_CHAN, axes=axes[0, j], show=False,
            title=cond.capitalize())
        grand[cond]["diff"].plot_topomap(
            times=[(REWP_TMIN + REWP_TMAX) / 2],
            average=REWP_TMAX - REWP_TMIN,
            axes=axes[1, j], show=False, colorbar=False)
    fig.savefig(deriv_root / "figure2_grand_average.png", dpi=200)
    plt.show()

    print(f"\n{len(all_measures)}/{len(subjects)} subjects retained")
    print(f"Outputs written to {deriv_root}")


if __name__ == "__main__":
    main()
