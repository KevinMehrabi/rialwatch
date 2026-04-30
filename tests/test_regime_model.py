import datetime as dt
import json
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest import mock

from scripts import pipeline


class RegimeModelTests(unittest.TestCase):
    def test_toman_to_rial_normalization(self) -> None:
        self.assertEqual(pipeline.normalize_unit(136_000.0, "toman"), 1_360_000.0)
        self.assertEqual(pipeline.normalize_unit(1_360_000.0, "rial"), 1_360_000.0)

    def test_gap_calculations(self) -> None:
        benchmark_results = {
            "open_market": {"fix": 1_600_000.0, "available": True},
            "official": {"fix": 1_380_000.0, "available": True},
            "regional_transfer": {"fix": 1_500_000.0, "available": True},
            "crypto_usdt": {"fix": 1_650_000.0, "available": True},
        }
        indicators = pipeline.compute_indicator_results(benchmark_results)
        self.assertAlmostEqual(indicators["street_official_gap_pct"]["value"], ((1_600_000 - 1_380_000) / 1_380_000) * 100)
        self.assertAlmostEqual(indicators["street_transfer_gap_pct"]["value"], ((1_600_000 - 1_500_000) / 1_500_000) * 100)
        self.assertAlmostEqual(indicators["street_crypto_gap_pct"]["value"], ((1_650_000 - 1_600_000) / 1_600_000) * 100)

    def test_official_sanity_warning(self) -> None:
        warnings = pipeline.evaluate_official_sanity_warnings(
            {
                "open_market": {"fix": 1_500_000.0},
                "official": {"fix": 2_100_000.0},
            }
        )
        self.assertTrue(warnings)
        self.assertTrue(any("above street_usd_irr" in msg for msg in warnings))

    def test_deprecated_fields_removed(self) -> None:
        self.assertNotIn("nima", pipeline.BENCHMARK_LABELS)
        self.assertNotIn("street_nima_gap", pipeline.INDICATOR_LABELS)
        self.assertNotIn("street_mobadeleh_gap", pipeline.INDICATOR_LABELS)

    def test_legacy_navasan_samples_apply_symbol_unit_map(self) -> None:
        sample = pipeline.parse_sample_record(
            "navasan",
            {
                "sampled_at": "2026-03-09T14:09:03Z",
                "value": 166400.0,
                "benchmarks": {
                    "open_market": 166400.0,
                    "official": 420000.0,
                    "regional_transfer": 169700.0,
                    "crypto_usdt": 149700.0,
                },
                "ok": True,
                "stale": False,
            },
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.source_unit, "mixed")
        self.assertEqual(sample.value, 1_664_000.0)
        self.assertEqual(sample.benchmark_values["open_market"], 1_664_000.0)
        self.assertEqual(sample.benchmark_values["official"], 420_000.0)

    def test_parse_source_payload_supports_navasan_js_assignment(self) -> None:
        body = (
            'var lastrates = {"mob_usd":{"value":"42,000"}};'
            'var yesterday = {"usd_sell":{"value":166400},"mex_usd_sell":{"value":1325072}};'
        )
        payload, mode = pipeline.parse_source_payload("navasan", body)
        self.assertEqual(mode, "javascript_var:yesterday")
        self.assertIn("mex_usd_sell", payload)
        self.assertEqual(payload["mex_usd_sell"]["value"], 1325072)

    def test_legacy_navasan_unknown_value_assumes_toman(self) -> None:
        sample = pipeline.parse_sample_record(
            "navasan",
            {
                "sampled_at": "2026-03-08T22:22:47Z",
                "value": 153500.0,
                "ok": True,
                "stale": False,
            },
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.value, 1_535_000.0)
        self.assertEqual(sample.source_unit, "mixed")

    def test_extract_value_by_symbol_candidates_supports_aux_p_field(self) -> None:
        payload = {
            "current": {
                "ice_transfer_usd_sell": {
                    "p": "1,403,083",
                    "ts": "2026-03-27 00:00:00",
                }
            }
        }
        value = pipeline.extract_value_by_symbol_candidates(payload, ("ice_transfer_usd_sell",))
        self.assertEqual(value, 1_403_083.0)

    def test_extract_benchmark_values_prefers_priority_symbol_order(self) -> None:
        payload = {
            "current": {
                "ice_currency_usd_sell": {
                    "p": "1,644,182",
                    "ts": "2026-04-13 00:00:00",
                },
                "ice_average_usd_sell": {
                    "p": "1,311,134",
                    "ts": "2026-04-13 00:00:00",
                },
            }
        }
        values, selected = pipeline.extract_benchmark_values_with_metadata(payload, "commercial_aux")
        self.assertEqual(values["official"], 1_644_182.0)
        self.assertEqual(selected.get("official"), "ice_currency_usd_sell")

    def test_extract_symbol_quote_time_supports_nested_symbol_paths(self) -> None:
        payload = {
            "current": {
                "ice_transfer_usd_sell": {
                    "p": "1,403,083",
                    "ts": "2026-03-27 00:00:00",
                }
            }
        }
        quote_time = pipeline.extract_symbol_quote_time(payload, ("ice_transfer_usd_sell",))
        self.assertIsNotNone(quote_time)
        assert quote_time is not None
        self.assertEqual(quote_time, dt.datetime(2026, 3, 27, 0, 0, tzinfo=dt.timezone.utc))

    def test_parse_tgju_profile_quote_time_from_jalali_history(self) -> None:
        body = """
        <table>
          <tr><td>1405/01/07</td><td>1,403,083</td></tr>
          <tr><td>1405/01/06</td><td>1,403,083</td></tr>
        </table>
        """
        parsed = pipeline.parse_tgju_profile_quote_time(body)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed, dt.datetime(2026, 3, 27, 8, 30, tzinfo=dt.timezone.utc))

    def test_parse_tgju_profile_current_value_from_table(self) -> None:
        body = """
        <table>
          <tr><td class="text-right">نرخ فعلی</td><td class="text-left">1,306,541</td></tr>
        </table>
        """
        parsed = pipeline.parse_tgju_profile_current_value(body)
        self.assertEqual(parsed, 1_306_541.0)

    def test_legacy_source_alias_maps_commercial_to_aux(self) -> None:
        self.assertEqual(pipeline.canonical_source_name("commercial"), "commercial_aux")

    def test_tgju_profile_url_for_symbol_rewrites_profile_path(self) -> None:
        base = "https://www.tgju.org/profile/sana_sell_usd"
        rewritten = pipeline.tgju_profile_url_for_symbol(base, "mex_usd_sell")
        self.assertEqual(rewritten, "https://www.tgju.org/profile/mex_usd_sell")

    def test_profile_source_switches_to_fresher_symbol_candidate(self) -> None:
        sampled_at = dt.datetime(2026, 4, 15, 14, 10, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 15, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 15, 14, 15, tzinfo=dt.timezone.utc)
        config = pipeline.SourceConfig(
            name="commercial_profile_sana",
            url="https://www.tgju.org/profile/mex_usd_sell",
            auth_mode="public_html",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        )

        stale_candidate = pipeline.Sample(
            source="commercial_profile_sana",
            sampled_at=sampled_at,
            value=1_401_825.0,
            benchmark_values={
                "open_market": None,
                "official": 1_401_825.0,
                "regional_transfer": None,
                "crypto_usdt": None,
                "emami_gold_coin": None,
            },
            quote_time=dt.datetime(2026, 3, 27, 8, 30, tzinfo=dt.timezone.utc),
            ok=False,
            stale=True,
            error="stale quote",
            health={"fetch_success": True, "request_url": "https://www.tgju.org/profile/sana_sell_usd"},
            source_unit="rial",
            normalized_unit="rial",
        )
        fresh_candidate = pipeline.Sample(
            source="commercial_profile_sana",
            sampled_at=sampled_at,
            value=1_401_825.0,
            benchmark_values={
                "open_market": None,
                "official": 1_401_825.0,
                "regional_transfer": None,
                "crypto_usdt": None,
                "emami_gold_coin": None,
            },
            quote_time=dt.datetime(2026, 4, 15, 11, 30, tzinfo=dt.timezone.utc),
            ok=False,
            stale=True,
            error="sample outside observation window",
            health={"fetch_success": True, "request_url": "https://www.tgju.org/profile/mex_usd_sell"},
            source_unit="rial",
            normalized_unit="rial",
        )

        with mock.patch(
            "scripts.pipeline.fetch_tgju_profile_official_public",
            side_effect=[stale_candidate, fresh_candidate],
        ) as mocked_profile_fetch:
            with mock.patch("scripts.pipeline.fetch_one") as mocked_fetch_one:
                sample = pipeline.fetch_tgju_profile_official_with_fallback(
                    config=config,
                    sampled_at=sampled_at,
                    window_start_dt=window_start,
                    window_end_dt=window_end,
                    profile_symbols=("sana_sell_usd", "mex_usd_sell"),
                )
        self.assertEqual(mocked_profile_fetch.call_count, 2)
        mocked_fetch_one.assert_not_called()
        self.assertEqual(sample.benchmark_values["official"], 1_401_825.0)
        self.assertEqual(sample.health.get("profile_symbol_selected"), "mex_usd_sell")

    def test_profile_source_uses_aux_fallback_when_profiles_fail(self) -> None:
        sampled_at = dt.datetime(2026, 4, 15, 14, 10, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 15, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 15, 14, 15, tzinfo=dt.timezone.utc)
        config = pipeline.SourceConfig(
            name="commercial_profile_sana",
            url="https://www.tgju.org/profile/mex_usd_sell",
            auth_mode="public_html",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        )

        failed_profile = pipeline.Sample(
            source="commercial_profile_sana",
            sampled_at=sampled_at,
            value=None,
            benchmark_values={
                "open_market": None,
                "official": None,
                "regional_transfer": None,
                "crypto_usdt": None,
                "emami_gold_coin": None,
            },
            quote_time=None,
            ok=False,
            stale=False,
            error="empty response body",
            health={"fetch_success": False, "failure_reason": "empty response body"},
            source_unit="rial",
            normalized_unit="rial",
        )
        aux_fallback = pipeline.Sample(
            source="commercial_aux",
            sampled_at=sampled_at,
            value=273_000.0,
            benchmark_values={
                "open_market": 273_000.0,
                "official": 1_401_825.0,
                "regional_transfer": None,
                "crypto_usdt": 273_000.0,
                "emami_gold_coin": 794_991_830.0,
            },
            quote_time=dt.datetime(2026, 4, 15, 11, 30, tzinfo=dt.timezone.utc),
            ok=False,
            stale=True,
            error="sample outside observation window",
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-04-15T11:30:00Z"},
                "request_url": "https://call.tgju.org/ajax.json",
            },
            source_unit="rial",
            normalized_unit="rial",
        )

        with mock.patch(
            "scripts.pipeline.fetch_tgju_profile_official_public",
            side_effect=[failed_profile, failed_profile],
        ):
            with mock.patch("scripts.pipeline.fetch_one", return_value=aux_fallback):
                sample = pipeline.fetch_tgju_profile_official_with_fallback(
                    config=config,
                    sampled_at=sampled_at,
                    window_start_dt=window_start,
                    window_end_dt=window_end,
                    profile_symbols=("sana_sell_usd", "mex_usd_sell"),
                )

        self.assertEqual(sample.source, "commercial_profile_sana")
        self.assertEqual(sample.benchmark_values["official"], 1_401_825.0)
        self.assertTrue(sample.health.get("fallback_used"))
        self.assertEqual(sample.health.get("fetch_mode"), "profile_fallback_aux")
        self.assertEqual(sample.health.get("fallback_reason"), "profile quote unavailable")
        self.assertEqual(len(sample.health.get("profile_attempts", [])), 2)

    def test_official_benchmark_prefers_freshest_source_quote(self) -> None:
        older = pipeline.Sample(
            source="navasan",
            sampled_at=dt.datetime(2026, 3, 29, 12, 10, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_325_072.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-27T00:00:00Z"},
            },
            source_unit="rial",
        )
        fresher = pipeline.Sample(
            source="commercial_aux_b",
            sampled_at=dt.datetime(2026, 3, 29, 12, 11, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_403_083.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:11:00Z"},
            },
            source_unit="rial",
        )
        samples = {
            "navasan": [older],
            "commercial_aux_b": [fresher],
        }
        benchmark_sources = {
            "navasan": ("official",),
            "commercial_aux_b": ("official",),
        }
        result = pipeline.compute_benchmark_result(samples, "official", benchmark_sources)
        self.assertEqual(result["fix"], 1_403_083.0)
        self.assertEqual(result["selection_method"], "freshest_quote_time")
        self.assertEqual(result["selected_sources"], ["commercial_aux_b"])
        self.assertEqual(result["source_update_counts"]["commercial_aux_b"], 1)
        self.assertFalse(result["withheld"])

    def test_official_benchmark_breaks_freshest_ties_by_update_cadence(self) -> None:
        navasan_only_latest = pipeline.Sample(
            source="navasan",
            sampled_at=dt.datetime(2026, 3, 29, 12, 11, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_325_072.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:11:00Z"},
            },
            source_unit="rial",
        )
        aux_older = pipeline.Sample(
            source="commercial_aux_c",
            sampled_at=dt.datetime(2026, 3, 29, 12, 9, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_402_500.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:09:00Z"},
            },
            source_unit="rial",
        )
        aux_latest = pipeline.Sample(
            source="commercial_aux_c",
            sampled_at=dt.datetime(2026, 3, 29, 12, 11, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_403_083.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:11:00Z"},
            },
            source_unit="rial",
        )
        samples = {
            "navasan": [navasan_only_latest],
            "commercial_aux_c": [aux_older, aux_latest],
        }
        benchmark_sources = {
            "navasan": ("official",),
            "commercial_aux_c": ("official",),
        }
        result = pipeline.compute_benchmark_result(samples, "official", benchmark_sources)
        self.assertEqual(result["fix"], 1_403_083.0)
        self.assertEqual(result["selection_method"], "freshest_quote_time_then_update_cadence")
        self.assertEqual(result["selected_sources"], ["commercial_aux_c"])
        self.assertEqual(result["source_update_counts"]["commercial_aux_c"], 2)
        self.assertEqual(result["source_update_counts"]["navasan"], 1)
        self.assertFalse(result["withheld"])

    def test_official_benchmark_uses_stale_fallback_when_only_stale_quote_exists(self) -> None:
        stale_quote = pipeline.Sample(
            source="commercial_aux",
            sampled_at=dt.datetime(2026, 4, 13, 14, 43, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_403_083.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-27T00:00:00Z"},
            },
            source_unit="rial",
        )
        samples = {
            "commercial_aux": [stale_quote],
        }
        benchmark_sources = {
            "commercial_aux": ("official",),
        }
        result = pipeline.compute_benchmark_result(samples, "official", benchmark_sources)
        self.assertFalse(result["withheld"])
        self.assertEqual(result["fix"], 1_403_083.0)
        self.assertTrue(result["available"])
        self.assertTrue(result["using_stale_fallback"])
        self.assertEqual(result["selected_sources"], ["commercial_aux"])
        self.assertIn("stale fallback quote", result["source_notes"]["commercial_aux"])

    def test_alanchand_companion_fallback_preserves_extracted_values(self) -> None:
        sampled_at = dt.datetime(2026, 4, 11, 14, 5, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 11, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 11, 14, 15, tzinfo=dt.timezone.utc)

        public_regional = {
            "value": 1_590_900.0,
            "raw_value": 1_590_900.0,
            "quote_time": dt.datetime(2026, 4, 11, 14, 3, tzinfo=dt.timezone.utc),
            "source_unit": "rial",
            "normalized_unit": "rial",
            "health": {"fetch_success": True},
        }
        public_crypto_missing = {
            "value": None,
            "raw_value": None,
            "quote_time": None,
            "source_unit": "rial",
            "normalized_unit": "rial",
            "health": {"fetch_success": False, "failure_reason": "http 503"},
        }
        navasan_sample = pipeline.Sample(
            source="navasan",
            sampled_at=sampled_at,
            value=1_664_000.0,
            benchmark_values={
                "open_market": 1_664_000.0,
                "official": None,
                "regional_transfer": None,
                "crypto_usdt": 1_587_500.0,
                "emami_gold_coin": None,
            },
            quote_time=sampled_at,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "raw_extracted_values": {"crypto_usdt": 158_750.0},
                "benchmark_quote_times": {"crypto_usdt": "2026-04-11T14:04:00Z"},
            },
            source_unit="mixed",
            normalized_unit="rial",
        )

        with mock.patch(
            "scripts.pipeline.fetch_alanchand_public_single_rate",
            side_effect=[public_regional, public_crypto_missing],
        ):
            with mock.patch("scripts.pipeline.fetch_one", return_value=navasan_sample):
                sample = pipeline.fetch_alanchand_companion_fallback(
                    sampled_at=sampled_at,
                    window_start_dt=window_start,
                    window_end_dt=window_end,
                    primary_error="http 401",
                    primary_error_type="http_error",
                )

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.benchmark_values["regional_transfer"], 1_590_900.0)
        self.assertEqual(sample.benchmark_values["crypto_usdt"], 1_587_500.0)
        self.assertEqual(sample.source_unit, "rial")
        self.assertEqual(sample.normalized_unit, "rial")
        health = sample.health or {}
        self.assertTrue(health.get("fetch_success"))
        self.assertIn("alanchand_public_pages", health.get("fallback_sources", []))
        self.assertIn("navasan", health.get("fallback_sources", []))
        self.assertEqual(health.get("extracted_values", {}).get("regional_transfer"), 1_590_900.0)
        self.assertEqual(health.get("extracted_values", {}).get("crypto_usdt"), 1_587_500.0)

    def test_fetch_one_navasan_uses_companion_fallback_after_http_429(self) -> None:
        config = pipeline.SourceConfig(
            name="navasan",
            url="https://api.navasan.tech/latest/",
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
            benchmark_families=("open_market", "regional_transfer", "crypto_usdt"),
            default_unit="toman",
        )
        sampled_at = dt.datetime(2026, 4, 11, 14, 0, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 11, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 11, 14, 15, tzinfo=dt.timezone.utc)
        retry_error = urllib.error.HTTPError(
            url="https://api.navasan.tech/latest/?api_key=***",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        fallback_sample = pipeline.Sample(
            source="navasan",
            sampled_at=sampled_at,
            value=None,
            benchmark_values={
                "open_market": None,
                "official": 1_401_825.0,
                "regional_transfer": 1_562_000.0,
                "crypto_usdt": 1_543_330.0,
                "emami_gold_coin": None,
            },
            quote_time=sampled_at,
            ok=False,
            stale=True,
            error="sample outside observation window",
            health={
                "fetch_success": True,
                "fallback_used": True,
                "fetch_mode": "companion_fallback",
            },
            source_unit="rial",
            normalized_unit="rial",
        )

        with mock.patch.dict(
            "os.environ",
            {
                "NAVASAN_API_KEY": "test-token",
                "API_SOURCE_RETRY_ATTEMPTS": "2",
                "API_SOURCE_RETRY_BACKOFF_SECONDS": "0",
            },
            clear=False,
        ):
            with mock.patch(
                "scripts.pipeline.fetch_request_body",
                side_effect=[retry_error],
            ):
                with mock.patch(
                    "scripts.pipeline.fetch_navasan_companion_fallback",
                    return_value=fallback_sample,
                ) as fallback_mock:
                    sample = pipeline.fetch_one(config, sampled_at, window_start, window_end)

        self.assertEqual(sample.benchmark_values["official"], 1_401_825.0)
        self.assertTrue(sample.health.get("fallback_used"))
        self.assertEqual(fallback_mock.call_count, 1)

    def test_build_source_configs_uses_public_navasan_site(self) -> None:
        with mock.patch.dict("os.environ", {"NAVASAN_PUBLIC_URL": "", "NAVASAN_SITE_URL": ""}, clear=False):
            configs = {cfg.name: cfg for cfg in pipeline.build_source_configs()}
        navasan = configs["navasan"]
        self.assertEqual(navasan.auth_mode, "public_html")
        self.assertEqual(navasan.secret_fields, ())
        self.assertEqual(navasan.url, pipeline.NAVASAN_PUBLIC_URL_DEFAULT)
        self.assertNotIn("official", navasan.benchmark_families)

    def test_fetch_one_navasan_public_website_combines_endpoint_payloads(self) -> None:
        config = pipeline.SourceConfig(
            name="navasan",
            url="https://www.navasan.net/",
            auth_mode="public_html",
            secret_fields=(),
            benchmark_families=("open_market", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
            default_unit="toman",
        )
        sampled_at = dt.datetime(2026, 4, 29, 14, 0, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 29, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 29, 14, 15, tzinfo=dt.timezone.utc)
        quote_ts = int(sampled_at.timestamp())
        endpoint_payloads = {
            "last_currencies": {"usd": {"value": 175_500, "date": quote_ts}},
            "aed_based_rates": {
                "usd_shakhs": {"value": 179_820, "date": quote_ts},
                "usd_usdt": {"value": 174_700, "date": quote_ts},
            },
            "gold_rates": {"sekkeh": {"value": 202_000_000, "date": quote_ts}},
        }

        def fake_endpoint(_base_url: str, endpoint_key: str, _endpoint_path: str):
            return endpoint_payloads[endpoint_key], {"fetch_success": True, "endpoint": endpoint_key}

        with mock.patch("scripts.pipeline.fetch_navasan_public_json_endpoint", side_effect=fake_endpoint):
            sample = pipeline.fetch_one(config, sampled_at, window_start, window_end)

        self.assertTrue(sample.health.get("fetch_success"))
        self.assertEqual(sample.value, 1_755_000.0)
        self.assertEqual(sample.benchmark_values["regional_transfer"], 1_798_200.0)
        self.assertEqual(sample.benchmark_values["crypto_usdt"], 1_747_000.0)
        self.assertIsNone(sample.benchmark_values["official"])
        self.assertEqual(sample.benchmark_values["emami_gold_coin"], 2_020_000_000.0)
        selected = sample.health.get("selected_symbol_by_benchmark")
        self.assertEqual(selected.get("open_market"), "usd")
        self.assertEqual(selected.get("crypto_usdt"), "usd_usdt")

    def test_fetch_one_redacts_query_secret_from_final_url(self) -> None:
        config = pipeline.SourceConfig(
            name="navasan",
            url="https://api.navasan.tech/latest/",
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
            benchmark_families=("official",),
            default_unit="toman",
        )
        sampled_at = dt.datetime(2026, 4, 28, 14, 0, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 28, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 28, 14, 15, tzinfo=dt.timezone.utc)
        body = '{"mex_usd_sell":{"value":1325072}}'
        final_url = "https://api.navasan.tech/latest/?api_key=super-secret-token"

        with mock.patch.dict("os.environ", {"NAVASAN_API_KEY": "super-secret-token"}, clear=False):
            with mock.patch("scripts.pipeline.fetch_request_body", return_value=(body, final_url)):
                sample = pipeline.fetch_one(config, sampled_at, window_start, window_end)

        health = sample.health or {}
        self.assertNotIn("super-secret-token", health.get("request_url") or "")
        self.assertNotIn("super-secret-token", health.get("final_url") or "")
        self.assertIn("api_key=%2A%2A%2A", health.get("final_url") or "")

    def test_public_json_sanitizer_redacts_nested_secret_urls(self) -> None:
        payload = {
            "sources": {
                "navasan": {
                    "samples": [
                        {
                            "health": {
                                "final_url": "https://api.navasan.tech/latest/?api_key=super-secret-token",
                                "request_urls": [
                                    "https://api.navasan.tech/latest/?api_key=super-secret-token&symbol=usd"
                                ],
                            }
                        }
                    ]
                }
            }
        }

        sanitized = pipeline.sanitize_public_json_payload(payload)
        rendered = str(sanitized)
        self.assertNotIn("super-secret-token", rendered)
        self.assertIn("api_key=%2A%2A%2A", rendered)

    def test_publish_latest_strips_public_commercial_unsupported_official_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            site_dir = Path(tmp_dir)
            daily = {
                "date": "2026-04-30",
                "as_of": "2026-04-30T21:30:55Z",
                "computed": {
                    "fix": 1_779_500.0,
                    "status": "Green",
                    "withheld": False,
                    "benchmarks": {
                        "open_market": {"value": 1_779_500.0, "available": True},
                        "official": {"value": 1_411_295.0, "available": True},
                    },
                },
                "benchmarks": {
                    "open_market": {
                        "fix": 1_779_500.0,
                        "withheld": False,
                        "available": True,
                        "source_units": {"bonbast": "rial"},
                    },
                    "official": {
                        "fix": 1_411_295.0,
                        "withheld": False,
                        "available": True,
                    },
                    "regional_transfer": {"fix": None, "withheld": True, "available": False},
                    "crypto_usdt": {"fix": None, "withheld": True, "available": False},
                    "emami_gold_coin": {"fix": None, "withheld": True, "available": False},
                },
                "indicators": {},
                "sources": {
                    "navasan": {
                        "samples": [
                            {
                                "sampled_at": "2026-04-30T14:51:18Z",
                                "fetch_success": True,
                                "benchmarks": {
                                    "open_market": 1_760_000.0,
                                    "official": 1_325_072.0,
                                    "regional_transfer": 1_803_400.0,
                                },
                                "health": {
                                    "source_fields": {"official": ["mex_usd_sell"], "open_market": ["usd"]},
                                    "selected_symbol_by_benchmark": {
                                        "official": "mex_usd_sell",
                                        "open_market": "usd",
                                    },
                                    "benchmark_quote_times": {
                                        "official": "2026-01-06T01:07:21Z",
                                        "open_market": "2026-04-30T14:35:51Z",
                                    },
                                    "extracted_values": {
                                        "official": 1_325_072.0,
                                        "open_market": 1_760_000.0,
                                    },
                                    "raw_extracted_values": {
                                        "official": 1_325_072.0,
                                        "open_market": 176_000.0,
                                    },
                                    "benchmark_units": {"official": "rial", "open_market": "toman"},
                                },
                            }
                        ],
                        "benchmark_medians": {"official": 1_325_072.0, "open_market": 1_760_000.0},
                    }
                },
            }

            pipeline.publish_latest(site_dir, daily)

            published = json.loads((site_dir / "api" / "latest.json").read_text(encoding="utf-8"))
            sample = published["sources"]["navasan"]["samples"][0]
            health = sample["health"]
            self.assertIsNone(sample["benchmarks"]["official"])
            self.assertNotIn("official", published["sources"]["navasan"]["benchmark_medians"])
            self.assertNotIn("official", health["source_fields"])
            self.assertNotIn("official", health["selected_symbol_by_benchmark"])
            self.assertNotIn("official", health["benchmark_quote_times"])
            self.assertNotIn("official", health["extracted_values"])
            self.assertNotIn("official", health["raw_extracted_values"])
            self.assertNotIn("official", health["benchmark_units"])
            self.assertEqual(published["benchmarks"]["official"]["fix"], 1_411_295.0)

    def test_navasan_companion_fallback_calls_public_single_rate_with_supported_signature(self) -> None:
        sampled_at = dt.datetime(2026, 4, 16, 14, 5, tzinfo=dt.timezone.utc)
        window_start = dt.datetime(2026, 4, 16, 13, 45, tzinfo=dt.timezone.utc)
        window_end = dt.datetime(2026, 4, 16, 14, 15, tzinfo=dt.timezone.utc)

        aux_sample = pipeline.Sample(
            source="commercial_aux",
            sampled_at=sampled_at,
            value=273_000.0,
            benchmark_values={
                "open_market": 273_000.0,
                "official": 1_401_825.0,
                "regional_transfer": None,
                "crypto_usdt": 273_000.0,
                "emami_gold_coin": 794_991_830.0,
            },
            quote_time=dt.datetime(2026, 4, 16, 11, 30, tzinfo=dt.timezone.utc),
            ok=True,
            stale=False,
            error=None,
            health={"fetch_success": True, "raw_extracted_values": {"official": 1_401_825.0}},
            source_unit="rial",
            normalized_unit="rial",
        )

        call_args = []

        def strict_single_rate(page_url: str, sampled_at: dt.datetime, minimum: float, maximum: float):
            call_args.append((page_url, sampled_at, minimum, maximum))
            return {
                "value": 1_560_000.0 if "usd-hav" in page_url else 1_540_000.0,
                "raw_value": 1_560_000.0 if "usd-hav" in page_url else 1_540_000.0,
                "quote_time": sampled_at,
                "source_unit": "rial",
                "normalized_unit": "rial",
                "health": {"fetch_success": True},
            }

        with mock.patch("scripts.pipeline.fetch_one", return_value=aux_sample):
            with mock.patch("scripts.pipeline.fetch_alanchand_public_single_rate", side_effect=strict_single_rate):
                sample = pipeline.fetch_navasan_companion_fallback(
                    sampled_at=sampled_at,
                    window_start_dt=window_start,
                    window_end_dt=window_end,
                    primary_error="http 429",
                    primary_error_type="http_error",
                )

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.benchmark_values["official"], 1_401_825.0)
        self.assertEqual(sample.benchmark_values["regional_transfer"], 1_560_000.0)
        self.assertEqual(sample.benchmark_values["crypto_usdt"], 1_540_000.0)
        self.assertEqual(len(call_args), 2)

    def test_build_source_configs_includes_multiple_aux_hosts(self) -> None:
        source_names = {cfg.name for cfg in pipeline.build_source_configs()}
        self.assertTrue({"commercial_aux_a", "commercial_aux_b", "commercial_aux_c", "commercial_aux"}.issubset(source_names))


if __name__ == "__main__":
    unittest.main()
