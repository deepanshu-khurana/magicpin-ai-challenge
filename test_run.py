#!/usr/bin/env python3
"""HTTP regression tests for the local merchant engagement bot."""

import json
import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from judge_simulator import BOT_URL as JUDGE_BOT_URL
from judge_simulator import BotClient


BASE_URL = os.environ.get("BOT_URL", JUDGE_BOT_URL)
DATASET_DIR = Path(__file__).parent / "dataset"


def load_dataset():
    categories = {}
    for path in (DATASET_DIR / "categories").glob("*.json"):
        with path.open(encoding="utf-8") as dataset_file:
            category = json.load(dataset_file)
        categories[category.get("slug", path.stem)] = category

    result = SimpleNamespace(categories=categories)
    for filename, collection, id_key in (
        ("merchants_seed.json", "merchants", "merchant_id"),
        ("customers_seed.json", "customers", "customer_id"),
        ("triggers_seed.json", "triggers", "id"),
    ):
        with (DATASET_DIR / filename).open(encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
        items = data.get(collection, data.get(collection.rstrip("s"), []))
        setattr(result, collection, {item[id_key]: item for item in items if id_key in item})
    return result


class BotApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = BotClient(BASE_URL)
        cls.dataset = load_dataset()

        health, error, _ = cls.client.healthz()
        if error or not health or health.get("status") != "ok":
            raise RuntimeError(
                f"Bot unavailable at {BASE_URL}: {error or health}"
            )

    @staticmethod
    def unique_id(prefix):
        return f"{prefix}_{uuid.uuid4().hex}"

    def push_context(self, scope, context_id, payload, version=1):
        data, error, _ = self.client.push_context(
            scope, context_id, version, payload
        )
        self.assertIsNone(error, f"{scope}/{context_id}: {error}")
        self.assertIsNotNone(data, f"{scope}/{context_id}: empty response")
        self.assertTrue(data.get("accepted"), f"{scope}/{context_id}: {data}")
        return data

    def reply(self, conversation_id, merchant_id, message, turn=2):
        data, error, _ = self.client.reply(
            conversation_id, merchant_id, message, turn
        )
        self.assertIsNone(error, f"{conversation_id}: {error}")
        self.assertIsNotNone(data, f"{conversation_id}: empty response")
        self.assertIn(data.get("action"), {"send", "wait", "end"}, data)
        return data

    def test_health_and_metadata(self):
        health, error, _ = self.client.healthz()
        self.assertIsNone(error)
        self.assertEqual(health.get("status"), "ok")

        metadata, error, _ = self.client.metadata()
        self.assertIsNone(error)
        self.assertTrue(metadata.get("team_name"))
        self.assertTrue(metadata.get("model"))

    def test_push_all_seed_contexts(self):
        records = []
        records.extend(
            ("category", context_id, payload)
            for context_id, payload in self.dataset.categories.items()
        )
        records.extend(
            ("merchant", context_id, payload)
            for context_id, payload in self.dataset.merchants.items()
        )
        records.extend(
            ("customer", context_id, payload)
            for context_id, payload in self.dataset.customers.items()
        )
        records.extend(
            ("trigger", context_id, payload)
            for context_id, payload in self.dataset.triggers.items()
        )

        self.assertGreaterEqual(len(self.dataset.merchants), 10)
        for scope, context_id, payload in records:
            with self.subTest(scope=scope, context_id=context_id):
                self.push_context(scope, context_id, payload)

        probe_id = self.unique_id("idempotency_probe")
        probe_payload = {"probe": True}
        self.push_context("category", probe_id, probe_payload)
        self.push_context("category", probe_id, {"probe": False})

    def test_tick_returns_valid_actions(self):
        for context_id, payload in self.dataset.categories.items():
            self.push_context("category", context_id, payload)
        for context_id, payload in self.dataset.merchants.items():
            self.push_context("merchant", context_id, payload)
        for context_id, payload in self.dataset.customers.items():
            self.push_context("customer", context_id, payload)

        source = next(iter(self.dataset.triggers.values()))
        trigger = json.loads(json.dumps(source))
        trigger_id = self.unique_id("test_trigger")
        trigger["id"] = trigger_id
        trigger["suppression_key"] = self.unique_id("test_suppression")
        self.push_context("trigger", trigger_id, trigger)

        data, error, _ = self.client.tick([trigger_id])
        self.assertIsNone(error)
        self.assertIsNotNone(data)
        self.assertIn("actions", data)
        self.assertEqual(len(data["actions"]), 1, data)
        action = data["actions"][0]
        for key in ("conversation_id", "merchant_id", "body", "cta", "trigger_id"):
            self.assertTrue(action.get(key), f"Action missing {key}: {action}")

    def test_engaged_reply_for_every_seed_merchant(self):
        self.assertGreaterEqual(len(self.dataset.merchants), 10)
        for merchant_id in self.dataset.merchants:
            with self.subTest(merchant_id=merchant_id):
                response = self.reply(
                    self.unique_id("merchant_engagement"),
                    merchant_id,
                    "I have a question about improving my business listing.",
                )
                self.assertEqual(response["action"], "send", response)
                self.assertTrue(response.get("body"), response)

    def test_auto_reply_detection(self):
        response = self.reply(
            self.unique_id("auto_reply"),
            next(iter(self.dataset.merchants)),
            "Thank you for contacting us! Our team will respond shortly.",
        )
        self.assertEqual(response["action"], "end", response)

    def test_intent_transition(self):
        response = self.reply(
            self.unique_id("intent"),
            next(iter(self.dataset.merchants)),
            "Ok lets do it. Whats next?",
        )
        self.assertEqual(response["action"], "send", response)
        self.assertIn("sending", response.get("body", "").lower())

    def test_hostile_opt_out(self):
        response = self.reply(
            self.unique_id("hostile"),
            next(iter(self.dataset.merchants)),
            "Stop messaging me. This is useless spam.",
        )
        self.assertEqual(response["action"], "end", response)

    def test_off_topic_question_is_redirected(self):
        response = self.reply(
            self.unique_id("off_topic"),
            next(iter(self.dataset.merchants)),
            "Can you help me with GST filing?",
        )
        self.assertEqual(response["action"], "send", response)
        self.assertIn("CA", response.get("body", ""))

    def test_repeated_reply_threshold(self):
        conversation_id = self.unique_id("repeated_reply")
        merchant_id = next(iter(self.dataset.merchants))
        message = "We received your note and will get back to you soon."
        actions = [
            self.reply(conversation_id, merchant_id, message, turn)["action"]
            for turn in (2, 3, 4)
        ]
        self.assertEqual(actions, ["send", "send", "end"])


if __name__ == "__main__":
    print(f"Testing bot at {BASE_URL}", flush=True)
    unittest.main(verbosity=2)