import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.pipeline.build_dataset import (
    PipelineConfig,
    load_product_category,
    load_ocr_detail,
    run_pipeline,
)
from analytics.pipeline.text_integrated_analysis import build_text_integrated_artifacts


class BuildDatasetPipelineTest(unittest.TestCase):
    def test_load_ocr_detail_normalizes_sparse_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp)
            payload = {
                "detail_02": {"full_text": "  첫 줄  \n\n첫 줄\n둘째   줄 "},
                "detail_05": {"full_text": "셋째 줄"},
                "detail_09": {"full_text": ""},
            }
            (product_dir / "ocr_data.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            detail = load_ocr_detail(product_dir)

            self.assertTrue(detail["ocr_source_exists"])
            self.assertTrue(detail["ocr_has_data"])
            self.assertEqual(detail["ocr_slot_count"], 2)
            self.assertEqual(detail["ocr_text_joined"], "첫 줄\n둘째 줄\n\n셋째 줄")
            self.assertIsNotNone(detail["ocr_text_hash"])

    def test_load_product_category_normalizes_failure_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            product_dir = session_dir / "products" / "2002"
            product_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "product_id": "2002",
                "status": "taxonomy_gap",
                "category_l1": "미분류",
                "category_l2": "미분류",
                "category_l3": "미분류",
                "category_code": "unknown",
                "confidence": 0.12,
                "review_reasons": ["taxonomy_gap_unknown"],
                "taxonomy_gap_candidate": "fashion.gift.card",
            }
            (product_dir / "category_result.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            category = load_product_category(session_dir, "2002", crawl_status="success")

            self.assertEqual(category.normalized_fields["category_payload_source"], "result_json")
            self.assertEqual(category.normalized_fields["category_ingest_status"], "failure")
            self.assertEqual(category.normalized_fields["category_quality_tier"], "none")
            self.assertEqual(category.normalized_fields["category_taxonomy_gap_candidate"], "fashion.gift.card")

    def test_run_pipeline_creates_dim_products_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "output"

            self._write_snapshot(
                data_dir,
                "2026-02-01",
                "09-00",
                [
                    {
                        "rank": 1,
                        "product_id": "1001",
                        "product_url": "https://example.com/products/1001",
                        "brand": "테스트브랜드",
                        "name": "테스트 팬츠 [블랙]",
                        "price": "39,900원",
                        "discount": "10%",
                        "vlm_raw_label": "팬츠",
                        "taxonomy_gap_candidate": "팬츠",
                        "crawl_status": "success",
                        "tags_count": 2,
                        "info_count": 2,
                        "images_count": 2,
                    },
                    {
                        "rank": 1,
                        "product_id": "1001",
                        "product_url": "https://example.com/products/1001",
                        "brand": "테스트브랜드",
                        "name": "테스트 팬츠 [블랙]",
                        "price": "39,900원",
                        "discount": "10%",
                        "vlm_raw_label": "팬츠",
                        "taxonomy_gap_candidate": "팬츠",
                        "crawl_status": "success",
                        "tags_count": 2,
                        "info_count": 2,
                        "images_count": 2,
                    },
                    {
                        "rank": 2,
                        "product_id": "9999",
                        "product_url": "https://example.com/products/9999",
                        "brand": "누락브랜드",
                        "name": "누락 상품",
                        "price": "19,000원",
                        "discount": None,
                        "crawl_status": "failed",
                        "error": "page load failed",
                    },
                ],
                product_payloads={
                    "1001": {
                        "tags": ["#와이드", "#데일리"],
                        "info": {"소재": "코튼 100%", "색상": "블랙"},
                        "ocr": {
                            "detail_01": {"full_text": "대표 설명 텍스트"},
                            "detail_05": {"full_text": "추가 설명"},
                        },
                    },
                },
                category_summary_rows=[
                    {
                        "product_id": "1001",
                        "snapshot_at": "09-00",
                        "category_l1": "패션",
                        "category_l2": "의류",
                        "category_l3": "슬랙스",
                        "category_code": "fashion.apparel.bottom.slacks",
                        "primary_color": "black",
                        "confidence": 0.98,
                        "status": "ok",
                        "decision_source": "fused",
                        "review_reasons": [],
                        "image_path": "/tmp/main.jpg",
                        "classified_at": "2026-02-01T09:00:00+00:00",
                    }
                ],
            )
            self._write_snapshot(
                data_dir,
                "2026-02-01",
                "14-00",
                [
                    {
                        "rank": 3,
                        "product_id": "1001",
                        "product_url": "https://example.com/products/1001",
                        "brand": "테스트브랜드",
                        "name": "테스트 팬츠 [블랙]",
                        "price": "35,900원",
                        "discount": "15%",
                        "vlm_raw_label": "팬츠",
                        "taxonomy_gap_candidate": "팬츠",
                        "crawl_status": "success",
                        "tags_count": 2,
                        "info_count": 2,
                        "images_count": 1,
                    },
                ],
                product_payloads={
                    "1001": {
                        "tags": ["#와이드", "#데일리"],
                        "info": {"소재": "코튼 100%", "색상": "블랙"},
                        "ocr": {
                            "detail_02": {"full_text": "최신 설명"}
                        },
                        "category": {
                            "product_id": "1001",
                            "snapshot_at": "14-00",
                            "category_l1": "패션",
                            "category_l2": "의류",
                            "category_l3": "슬랙스",
                            "category_code": "fashion.apparel.bottom.slacks",
                            "primary_color": "black",
                            "confidence": 0.97,
                            "status": "ok",
                            "decision_source": "image",
                            "review_reasons": [],
                            "image_path": "/tmp/secondary.jpg",
                            "classified_at": "2026-02-01T14:00:00+00:00",
                        },
                    },
                },
            )

            config = PipelineConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                enable_multimodal=False,
                enable_embeddings=False,
                enable_qdrant_upsert=False,
            )
            featured = run_pipeline(config)

            self.assertEqual(len(featured), 3)

            fact_df = pd.read_parquet(output_dir / "fact_snapshots.parquet")
            dim_df = pd.read_parquet(output_dir / "dim_products.parquet")
            coverage_df = pd.read_parquet(output_dir / "product_snapshot_coverage.parquet")
            raw_schema = json.loads((output_dir / "schema_raw.json").read_text(encoding="utf-8"))

            self.assertEqual(len(fact_df), 3)
            self.assertEqual(len(dim_df), 2)
            self.assertEqual(len(coverage_df), 3)
            self.assertNotIn("ocr_entries_json", fact_df.columns)
            self.assertNotIn("ocr_text_joined", fact_df.columns)
            self.assertIn("category_code", fact_df.columns)
            self.assertIn("category_ingest_status", fact_df.columns)
            raw_fields = {field["field"] for field in raw_schema.get("fields", [])}
            self.assertIn("product.vlm_raw_label", raw_fields)
            self.assertNotIn("product.taxonomy_gap_candidate", raw_fields)
            self.assertIn("category.category_code", raw_fields)

            product_1001 = dim_df[dim_df["product_id"] == "1001"].iloc[0]
            self.assertEqual(product_1001["representative_snapshot_id"], "data:2026-02-01_14-00")
            self.assertEqual(product_1001["ocr_text_joined"], "최신 설명")
            self.assertEqual(int(product_1001["observed_snapshot_count"]), 2)
            self.assertTrue(bool(product_1001["is_repeated_product"]))
            self.assertEqual(product_1001["category_code"], "fashion.apparel.bottom.slacks")
            self.assertEqual(product_1001["category_payload_source"], "result_json")
            self.assertEqual(product_1001["category_ingest_status"], "success")

            missing_row = coverage_df[coverage_df["product_id"] == "9999"].iloc[0]
            self.assertFalse(bool(missing_row["product_dir_exists"]))
            self.assertFalse(bool(missing_row["ocr_has_data"]))
            missing_fact = fact_df[fact_df["product_id"] == "9999"].iloc[0]
            self.assertEqual(missing_fact["category_ingest_status"], "skipped")
            self.assertEqual(missing_fact["category_skip_reason"], "crawl_failed")

    def test_text_integrated_artifacts_build_review_claim_gap_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviews_dir = root / "reviews" / "products" / "1001"
            reviews_dir.mkdir(parents=True, exist_ok=True)
            review_rows = [
                {
                    "review_id": 1,
                    "product_id": "1001",
                    "rating": 5,
                    "review_type": "general",
                    "photo_review": True,
                    "helpful_count": 2,
                    "created_at": "2026-04-14T10:00:00+09:00",
                    "review_text": "디자인이 예쁘고 착용감이 편해요",
                },
                {
                    "review_id": 2,
                    "product_id": "1001",
                    "rating": 3,
                    "review_type": "style",
                    "photo_review": False,
                    "helpful_count": 0,
                    "created_at": "2026-04-14T11:00:00+09:00",
                    "review_text": "사이즈가 조금 작고 가격이 아쉬워요",
                },
            ]
            (reviews_dir / "master.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in review_rows),
                encoding="utf-8",
            )

            featured_df = pd.DataFrame(
                [
                    {
                        "snapshot_id": "data:2026-04-14_01-50",
                        "snapshot_date": "2026-04-14",
                        "snapshot_time": "01-50",
                        "crawl_datetime": "2026-04-14T01:50:00",
                        "product_id": "1001",
                        "brand": "테스트브랜드",
                        "name": "테스트 슬랙스 [블랙]",
                        "category_code": "fashion.apparel.bottom.slacks",
                        "category_l1": "패션",
                        "category_l2": "의류",
                        "category_l3": "슬랙스",
                        "price_band": "3-7만",
                        "rank": 12,
                        "material": "코튼 100%",
                        "color": "블랙",
                        "tags_joined": "#데일리,#와이드",
                        "ocr_text_joined": "편안한 착용감과 깔끔한 디자인의 데일리 슬랙스",
                    }
                ]
            )
            featured_df["crawl_datetime"] = pd.to_datetime(featured_df["crawl_datetime"])

            artifacts = build_text_integrated_artifacts(featured_df, root / "reviews")

            review_facts = artifacts["text_review_facts"]
            claim_facts = artifacts["text_claim_facts"]
            gap_metrics = artifacts["text_gap_metrics"]

            self.assertFalse(review_facts.empty)
            self.assertFalse(claim_facts.empty)
            self.assertFalse(gap_metrics.empty)
            self.assertIn("aspect", review_facts.columns)
            self.assertIn("sort_source", review_facts.columns)
            self.assertIn("claim_score", claim_facts.columns)
            self.assertIn("gap_score", gap_metrics.columns)
            self.assertTrue((review_facts["product_id"] == "1001").all())
            self.assertGreaterEqual(int(gap_metrics["review_sentence_count"].sum()), 2)

    def _write_snapshot(
        self,
        data_dir: Path,
        snapshot_date: str,
        snapshot_time: str,
        products: list[dict],
        product_payloads: dict[str, dict],
        category_summary_rows: list[dict] | None = None,
    ) -> None:
        session_dir = data_dir / snapshot_date / snapshot_time
        products_dir = session_dir / "products"
        products_dir.mkdir(parents=True, exist_ok=True)
        ranking_summary = {
            "crawl_datetime": f"{snapshot_date}T{snapshot_time.replace('-', ':')}:00.000000",
            "total_products": len(products),
            "success_count": sum(1 for p in products if p.get("crawl_status") == "success"),
            "failed_count": sum(1 for p in products if p.get("crawl_status") != "success"),
            "products": products,
        }
        (session_dir / "ranking_summary.json").write_text(
            json.dumps(ranking_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if category_summary_rows:
            (session_dir / "category_summary.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in category_summary_rows),
                encoding="utf-8",
            )

        for product_id, payload in product_payloads.items():
            product_dir = products_dir / product_id
            product_dir.mkdir(parents=True, exist_ok=True)

            tags = payload.get("tags")
            if tags:
                pd.DataFrame({"태그": tags}).to_csv(
                    product_dir / "tags.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            info = payload.get("info")
            if info:
                pd.DataFrame(
                    [{"항목": key, "내용": value} for key, value in info.items()]
                ).to_csv(
                    product_dir / "product_info.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            ocr = payload.get("ocr")
            if ocr:
                (product_dir / "ocr_data.json").write_text(
                    json.dumps(ocr, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            category = payload.get("category")
            if category:
                (product_dir / "category_result.json").write_text(
                    json.dumps(category, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    unittest.main()
