/usr/local/lib/python3.11/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.8.0) or chardet (7.7.0)/charset_normalizer (3.5.2) doesn't match a supported version!
  warnings.warn(
test_duration (test_core.ConfigTests.test_duration) ... ok
test_environment_and_query_delimiter (test_core.ConfigTests.test_environment_and_query_delimiter) ... ok
test_rolling_bound (test_core.ConfigTests.test_rolling_bound) ... ok
test_signature_order_matters (test_core.ConfigTests.test_signature_order_matters) ... ok
test_bad_timestamps_and_duplicate_series (test_core.DataTests.test_bad_timestamps_and_duplicate_series) ... ok
test_chunked_alignment_and_schema (test_core.DataTests.test_chunked_alignment_and_schema) ... ok
test_extra_features_do_not_change_schema (test_core.DataTests.test_extra_features_do_not_change_schema) ... ok
test_feature_and_row_limits (test_core.DataTests.test_feature_and_row_limits) ... ok
test_forward_fill_only (test_core.DataTests.test_forward_fill_only) ... ok
test_http_warning_matrix_and_size_checks (test_core.DataTests.test_http_warning_matrix_and_size_checks) ... ok
test_invalid_shape_and_overflow_rejected (test_core.DataTests.test_invalid_shape_and_overflow_rejected) ... ok
test_missing_budget (test_core.DataTests.test_missing_budget) ... ok
test_missing_feature_is_not_zero (test_core.DataTests.test_missing_feature_is_not_zero) ... ok
test_missing_latest_online_rejected (test_core.DataTests.test_missing_latest_online_rejected) ... ok
test_atomic_json (test_core.LifecycleTests.test_atomic_json) ... ok
test_candidate_restore_verified_before_promotion (test_core.LifecycleTests.test_candidate_restore_verified_before_promotion) ... /usr/local/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:64: PyparsingDeprecationWarning: 'oneOf' deprecated - use 'one_of'
  prop = Group((name + Suppress("=") + comma_separated(value)) | oneOf(_CONSTANTS))
/usr/local/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85: PyparsingDeprecationWarning: 'parseString' deprecated - use 'parse_string'
  parse = parser.parseString(pattern)
/usr/local/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89: PyparsingDeprecationWarning: 'resetCache' deprecated - use 'reset_cache'
  parser.resetCache()
/usr/local/lib/python3.11/site-packages/matplotlib/_mathtext.py:45: PyparsingDeprecationWarning: 'enablePackrat' deprecated - use 'enable_packrat'
  ParserElement.enablePackrat()
In /usr/local/lib/python3.11/site-packages/matplotlib/mpl-data/stylelib/classic.mplstyle: 'parseString' deprecated - use 'parse_string'
In /usr/local/lib/python3.11/site-packages/matplotlib/mpl-data/stylelib/classic.mplstyle: 'resetCache' deprecated - use 'reset_cache'
ok
test_failed_training_removes_candidate_not_previous (test_core.LifecycleTests.test_failed_training_removes_candidate_not_previous) ... ok
test_subprocess_timeout_is_passed (test_core.LifecycleTests.test_subprocess_timeout_is_passed) ... ok
test_train_persist_restore_infer_contract (test_core.LifecycleTests.test_train_persist_restore_infer_contract) ... ok
test_fail_closed_and_staleness (test_core.MetricsTests.test_fail_closed_and_staleness) ... ok
test_malformed_result_is_not_ready (test_core.MetricsTests.test_malformed_result_is_not_ready) ... ok
test_training_failure_does_not_remove_current_result (test_core.MetricsTests.test_training_failure_does_not_remove_current_result) ... ok
test_actual_vendored_spot (test_core.POTTests.test_actual_vendored_spot) ... ok
test_invalid_or_degenerate_tail (test_core.POTTests.test_invalid_or_degenerate_tail) ... ok
test_scaler_dimension_and_nan (test_core.PreprocessTests.test_scaler_dimension_and_nan) ... ok
test_scaler_roundtrip_and_constant_column (test_core.PreprocessTests.test_scaler_roundtrip_and_constant_column) ... ok
test_window_alignment (test_core.PreprocessTests.test_window_alignment) ... ok
test_full_http_collection_train_restore_export_and_data_failure (test_http.HTTPIntegrationTests.test_full_http_collection_train_restore_export_and_data_failure) ... ok
test_real_http_dataframe_and_ttl (test_http.HTTPIntegrationTests.test_real_http_dataframe_and_ttl) ... ok

----------------------------------------------------------------------
Ran 29 tests in 8.768s

OK
