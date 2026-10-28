# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd

# ------------------------------------------------------------
# Paths (robust no matter where you call it from)
# ------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FUNCTIONS_DIR = os.path.join(ROOT_DIR, "functions")

# Make sure your local modules are importable
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ------------------------------------------------------------
# Imports from your repo
# ------------------------------------------------------------
from ML_Pipeline_CLEAR import MatchConfig, run_pipeline  # comes from functions/ML_Pipeline_CLEAR.py
from server_config import DATA_PATH                      # your central data dir


# ------------------------------------------------------------
# Default data file (as you described)
# ------------------------------------------------------------
DEFAULT_DATA_FILE = os.path.join(DATA_PATH, "toni_preprocessed.csv")


def build_cfg(models=None) -> MatchConfig:
    """
    Define your MatchConfig HERE (same as in your notebook),
    so the script is fully self-contained.
    """
    cfg = MatchConfig(
        PATIENT_ID_COL="patient_id",
        THERAPIST_ID_COL="therapist_id",
        DATE_COL="u_registrationDate_p",
        DATE_DAYFIRST=True,
        DATE_FORMAT=None,

        LABEL_COL="t1_distress_sum",
        STANDARDIZE_Y=True,

        # you can override RUN_MODELS via CLI (--models cat ridge ...)
        RUN_MODELS=models if models is not None else ["cat"],

        CAT_USE_NATIVE_CATEGORICAL=True,
        CAT_MIN_FREQ_FOR_OHE=1,

        # Make output dirs stable relative to repo root (recommended)
        OUTPUT_BASE_DIR=os.path.join(ROOT_DIR, "model_outputs"),
        CACHE_DIR=os.path.join(ROOT_DIR, ".sk_cache"),

        # ------------------------------------------------------------
        # Paste your feature lists from the notebook here
        # ------------------------------------------------------------
        patient_numeric=['t0_ses_p','surveys.t0.answers.1585.discrimination_1','surveys.t0.answers.1585.discrimination_2',
    'surveys.t0.answers.1585.discrimination_3','surveys.t0.answers.1585.discrimination_4','surveys.t0.answers.1585.discrimination_5',
    't0_phq_p_1','t0_phq_p_2','t0_phq_p_3','t0_phq_p_4','t0_phq_p_5','t0_phq_p_6','t0_phq_p_7','t0_phq_p_8',
    't0_gad_p_1','t0_gad_p_2','t0_gad_p_3','t0_gad_p_4','t0_gad_p_5','t0_gad_p_6','t0_gad_p_7',
    't0_swls_p_1','t0_swls_p_2','t0_swls_p_3','t0_swls_p_4','t0_swls_p_5',
    't0_aqol_p_1','t0_aqol_p_2','t0_aqol_p_3','t0_aqol_p_4','t0_aqol_p_5','t0_aqol_p_6','t0_aqol_p_7','t0_aqol_p_8','t0_aqol_p_9','t0_aqol_p_10',
    't0_aqol_p_11','t0_aqol_p_12','t0_aqol_p_13','t0_aqol_p_14','t0_aqol_p_15','t0_aqol_p_16','t0_aqol_p_17',
    't0_aqol_p_18','t0_aqol_p_19','t0_aqol_p_20','t0_aqol_p_21','t0_aqol_p_22','t0_aqol_p_23','t0_aqol_p_24','t0_aqol_p_25','t0_aqol_p_26',
    't0_aqol_p_27','t0_aqol_p_28','t0_aqol_p_29','t0_aqol_p_30','t0_aqol_p_31','t0_aqol_p_32','t0_aqol_p_33','t0_aqol_p_34','t0_aqol_p_35',
    't0_audit_p_1','t0_audit_p_2','t0_audit_p_3',
    't0_dudit_p_1','t0_dudit_p_2','t0_dudit_p_3','t0_dudit_p_4',
    't0_msfq_p_1','t0_msfq_p_2','t0_msfq_p_3','t0_msfq_p_4','t0_msfq_p_5',
    't0_pid_p_1','t0_pid_p_2','t0_pid_p_3','t0_pid_p_4','t0_pid_p_5','t0_pid_p_6','t0_pid_p_7','t0_pid_p_8','t0_pid_p_9','t0_pid_p_10',
    't0_pid_p_11','t0_pid_p_12','t0_pid_p_13','t0_pid_p_14','t0_pid_p_15','t0_pid_p_16','t0_pid_p_17','t0_pid_p_18','t0_pid_p_19','t0_pid_p_20',
    't0_pid_p_21','t0_pid_p_22','t0_pid_p_23','t0_pid_p_24','t0_pid_p_25','t0_pid_p_26','t0_pid_p_27','t0_pid_p_28','t0_pid_p_29','t0_pid_p_30',
    't0_pid_p_31','t0_pid_p_32','t0_pid_p_33','t0_pid_p_34','t0_pid_p_35','t0_pid_p_36',
    't0_opd_p_1','t0_opd_p_2','t0_opd_p_3','t0_opd_p_4','t0_opd_p_5','t0_opd_p_6','t0_opd_p_7','t0_opd_p_8','t0_opd_p_9','t0_opd_p_10','t0_opd_p_11','t0_opd_p_12',
    't0_ace_p_1','t0_ace_p_2','t0_ace_p_3','t0_ace_p_4','t0_ace_p_5','t0_ace_p_6','t0_ace_p_7','t0_ace_p_8','t0_ace_p_9','t0_ace_p_10',
    't0_ask_p_1','t0_ask_p_2','t0_ask_p_3',
    't0_mhse_p_1','t0_mhse_p_2','t0_mhse_p_3','t0_mhse_p_4','t0_mhse_p_5','t0_mhse_p_6',
    't0_change_p_1','t0_change_p_2','t0_dropout_p',
    't0_phq8_p_sum','t0_gad7_p_sum','t0_swls_p_sum',
    't0_aqol_p_coping','t0_aqol_p_relationship','t0_aqol_p_independent_living','t0_aqol_p_mental_health',
    't0_aqol_p_pain','t0_aqol_p_self_worth','t0_aqol_p_happiness','t0_aqol_p_senses',
    't0_audit_p_sum','t0_dudit_p_sum','t0_msfq_p_sum',
    't0_pid_p_negative_affect','t0_pid_p_antagonism','t0_pid_p_disinhibition','t0_pid_p_detachment','t0_pid_p_psychoticism','t0_pid_p_anankastia',
    't0_opd_p_self','t0_opd_p_contact','t0_opd_p_relationship',
    't0_ace_p_sum','t0_ask_p_sum','t0_mhse_p_sum','t0_age_p'
        ],
        patient_categorical=['t0_gender_p','t0_school_p','t0_employment_p','t0_location_p',
        't0_therapy_p_1','t0_therapy_p_2','t0_familyhistory_p','t0_meds_p',
        't0_relationship_p','t0_migration_p_1','t0_migration_p_2','t0_migration_p_3',
        't0_disease_p_A','t0_disease_p_B','t0_disease_p_C','t0_disease_p_D',
        't0_disease_p_E','t0_disease_p_F','t0_disease_p_G',
        ],
        therapist_numeric=['t0_age_t','t0_experience_t','t0_bfi_t_1','t0_bfi_t_2','t0_bfi_t_3','t0_bfi_t_4','t0_bfi_t_5',
        't0_bfi_t_6','t0_bfi_t_7','t0_bfi_t_8','t0_bfi_t_9','t0_bfi_t_10',
        't0_bfi_t_extraversion','t0_bfi_t_agreeableness','t0_bfi_t_conscientiousness',
        't0_bfi_t_neuroticism','t0_bfi_t_openness'
        ],
        therapist_categorical=['t0_method_t_CBT','t0_method_t_PDT','t0_method_t_ST','t0_method_t_AP',
        't0_appliedmethod_t_VT','t0_appliedmethod_t_TP','t0_appliedmethod_t_AP',
        't0_appliedmethod_t_ST','t0_appliedmethod_t_Other',
        't0_location_t','t0_gender_t','t0_relationship_t','t0_sexualorientation_t','t0_children_t'
        ],
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run TONI matching pipeline from terminal.")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE,
                        help=f"Path to input CSV (default: {DEFAULT_DATA_FILE})")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override cfg.RUN_MODELS, e.g. --models cat ridge fm ebm")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable writing outputs (cfg.OUTPUT_SAVE=False)")
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    print(f"[INFO] Loading data: {args.data_file}")
    df_all = pd.read_csv(args.data_file)

    # Build config (in this script)
    cfg = build_cfg(models=args.models)

    if args.no_save:
        cfg.OUTPUT_SAVE = False

    # Run pipeline
    out = run_pipeline(df_all, cfg)

    # Print summary + output dir
    print("\n==================== RUN COMPLETE ====================")
    if "output_dir" in out:
        print(f"Output directory: {out['output_dir']}")
    if "best_model_name" in out:
        print(f"Best model: {out['best_model_name']}")

    summary = out.get("summary", None)
    if isinstance(summary, pd.DataFrame):
        print("\n====================== SUMMARY =======================")
        with pd.option_context("display.width", 180, "display.max_columns", 300):
            print(summary.to_string(index=False))
    else:
        print("No summary DataFrame found. Output keys:", list(out.keys()))


if __name__ == "__main__":
    main()
