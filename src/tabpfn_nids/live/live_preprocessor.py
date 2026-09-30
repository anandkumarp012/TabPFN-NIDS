"""Live preprocessing: loads trained artifacts and transforms live features.

This module bridges the live pipeline to the existing preprocessing artifacts
(trained scaler, feature schema) so that live traffic goes through exactly
the same transformation chain as offline PCAP traffic.

IMPORTANT: The scaler is loaded ONCE at startup. It is NEVER re-fitted on
live data. This preserves the training-domain distribution.

The feature extraction path is:
    FlowRecord list
        → flows_to_dataframe()   [reuses existing flow_builder helper]
        → compute_all_features() [exact same as offline pipeline]
        → clean_data()           [same cleaner as offline]
        → select features by schema
        → apply fitted scaler
        → numpy array for TabPFN
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabpfn_nids.config import PROJECT_ROOT
from tabpfn_nids.features.feature_pipeline import compute_all_features
from tabpfn_nids.flows.flow_builder import FlowRecord, flows_to_dataframe
from tabpfn_nids.preprocessing.cleaner import clean_data

logger = logging.getLogger(__name__)

# Default artifact paths (same as analyze_pcap.py)
_DEFAULT_SCALER_PATH = (
    PROJECT_ROOT / "data" / "artifacts" / "preprocessing" / "fitted_scaler.pkl"
)
_DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "data" / "artifacts" / "models" / "model_feature_schema.json"
)


class ArtifactLoadError(RuntimeError):
    """Raised when a required training artifact cannot be loaded."""
    pass


class LivePreprocessor:
    """Loads training artifacts and transforms live flow features.

    Args:
        scaler_path: Path to the fitted scaler pickle.
        feature_schema_path: Path to the feature schema JSON.
    """

    def __init__(
        self,
        scaler_path: Path | str | None = None,
        feature_schema_path: Path | str | None = None,
    ) -> None:
        self._scaler_path = Path(scaler_path) if scaler_path else _DEFAULT_SCALER_PATH
        self._schema_path = (
            Path(feature_schema_path)
            if feature_schema_path
            else _DEFAULT_SCHEMA_PATH
        )
        self._scaler: Any = None
        self._feature_names: list[str] = []
        self._loaded = False

    def load(self) -> LivePreprocessor:
        """Load scaler and feature schema from disk.

        Returns:
            self for chaining.

        Raises:
            ArtifactLoadError: If files are missing or unreadable.
        """
        # Load feature schema
        if not self._schema_path.is_file():
            raise ArtifactLoadError(
                f"Feature schema not found at {self._schema_path}. "
                "Run the training pipeline (scripts/analyze_pcap.py) to "
                "generate preprocessing artifacts."
            )
        try:
            with open(self._schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            self._feature_names = schema.get("feature_names", [])
            if not self._feature_names:
                raise ArtifactLoadError(
                    f"Feature schema at {self._schema_path} has no 'feature_names'."
                )
            logger.info(
                "Feature schema loaded: %d features from %s",
                len(self._feature_names),
                self._schema_path,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ArtifactLoadError(
                f"Failed to parse feature schema at {self._schema_path}: {exc}"
            ) from exc

        # Load scaler
        if not self._scaler_path.is_file():
            raise ArtifactLoadError(
                f"Fitted scaler not found at {self._scaler_path}. "
                "Run the training pipeline to generate preprocessing artifacts."
            )
        try:
            with open(self._scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("Fitted scaler loaded from %s", self._scaler_path)
        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to load scaler from {self._scaler_path}: {exc}"
            ) from exc

        self._loaded = True
        return self

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names expected by the model."""
        return list(self._feature_names)

    def transform(
        self,
        flows: list[FlowRecord],
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Transform a list of FlowRecord objects into model-ready feature matrix.

        Uses the SAME compute_all_features() function as the offline pipeline
        to guarantee feature parity.

        Args:
            flows: Completed flow records from LiveFlowSession.

        Returns:
            Tuple of (X_scaled: np.ndarray, metadata: pd.DataFrame).
            X_scaled is ready for TabPFN.predict_proba().
            metadata contains flow_id, src_ip, dst_ip, etc.

        Raises:
            RuntimeError: If load() has not been called.
            ValueError: If feature validation fails.
        """
        if not self._loaded:
            raise RuntimeError(
                "LivePreprocessor.load() must be called before transform()."
            )

        if not flows:
            raise ValueError("No flows provided for transformation.")

        # Step 1: Convert FlowRecord list → DataFrame (same as offline)
        raw_df = flows_to_dataframe(flows)
        logger.debug(
            "Converting %d flows to feature DataFrame", len(raw_df)
        )

        # Step 2: Feature engineering (exact same as offline PCAP pipeline)
        feature_df = compute_all_features(raw_df)

        # Step 3: Basic cleaning (handle NaN/Inf from degenerate flows)
        feature_df, _ = clean_data(
            feature_df,
            missing_strategy="zero",  # live traffic can't wait for median
            remove_impossible=True,
            remove_duplicates=False,   # keep all flows, even near-duplicates
        )

        # Step 4: Validate feature names
        missing = [f for f in self._feature_names if f not in feature_df.columns]
        if missing:
            raise ValueError(
                f"Live feature extraction is missing {len(missing)} columns "
                f"expected by the model: {missing[:10]}...\n"
                "This indicates a mismatch between the live pipeline and the "
                "trained model. Inspect feature_pipeline.py and the model schema."
            )

        # Step 5: Extract metadata (identifiers, not ML features)
        metadata_cols = [
            c for c in feature_df.columns if c not in self._feature_names
        ]
        metadata = feature_df[metadata_cols].copy() if metadata_cols else pd.DataFrame()

        # Step 6: Select features in the exact schema order
        X = feature_df[self._feature_names].copy()

        # Step 7: Ensure numeric and handle any remaining issues
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)

        # Step 8: Final validation — TabPFN rejects non-finite inputs
        if not np.isfinite(X.values).all():
            n_bad = int((~np.isfinite(X.values)).sum())
            logger.error(
                "Live preprocessing: %d non-finite values remain after cleaning; "
                "replacing with 0.",
                n_bad,
            )
            X = X.fillna(0.0)

        # Step 9: Apply the SAME fitted scaler used in offline inference
        X_scaled = self._scaler.transform(X.values.astype(np.float64))

        logger.debug(
            "Preprocessed %d flows → X shape %s",
            len(flows),
            X_scaled.shape,
        )
        return X_scaled, metadata

    def is_loaded(self) -> bool:
        """Return True if artifacts are loaded and ready."""
        return self._loaded
