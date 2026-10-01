import copy
import io
import json
import os
import re
import runpy
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from bson import BSON, ObjectId
from bson.codec_options import CodecOptions
import langchain_groq
import pymongo

from app.data.restaurant_info import RESTAURANT_INFO
from pymongo.errors import PyMongoError

from app.tools.menu import MenuDataError, get_menu


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class DeterministicLLM:
    def _customer_message(self, prompt: str, stop_marker: str) -> str:
        return (
            prompt.split("Customer message:", 1)[1]
            .split(stop_marker, 1)[0]
            .strip()
        )

    def _extract_order_items(self, message: str) -> dict:
        normalized = message.casefold()

        if "dragon bowl" in normalized:
            items = [
                {"name": "Dragon Bowl", "quantity": 1},
                {
                    "name": "Cheese Veg Burger",
                    "quantity": 1,
                    "item_id": "untrusted-id",
                    "price": 1,
                },
            ]
            return {"items": items}

        if "cheese veg burger" in normalized and "paneer" in normalized:
            return {
                "items": [
                    {
                        "name": "Cheese Veg Burger",
                        "quantity": 2,
                        "item_id": "untrusted-id",
                        "price": 1,
                    },
                    {"name": "Paneer Burger", "quantity": 1},
                ]
            }

        quantity = 1
        if "-1" in normalized:
            quantity = -1
        elif re.search(r"\b0\b", normalized):
            quantity = 0
        elif re.search(r"\b2\b", normalized):
            quantity = 2

        if "cheese veg burger" in normalized:
            item_name = "Cheese Veg Burger"
        elif "paneer" in normalized:
            item_name = "Paneer Burger"
        else:
            item_name = "Dragon Bowl"

        return {
            "items": [
                {
                    "name": item_name,
                    "quantity": quantity,
                    "item_id": "untrusted-id",
                    "price": 1,
                }
            ]
        }

    def invoke(self, prompt: str) -> FakeLLMResponse:
        normalized_prompt = prompt.casefold()

        if "restaurant request classifier" in normalized_prompt:
            message = self._customer_message(prompt, "\n\nReply with ONLY:")
            if "binary search" in message.casefold():
                return FakeLLMResponse("OUT_OF_SCOPE")
            return FakeLLMResponse("RESTAURANT")

        if "intent classifier for a restaurant ai agent" in normalized_prompt:
            message = self._customer_message(prompt, "\n\nReply with ONLY")
            normalized_message = message.casefold()
            if "status" in normalized_message:
                intent = "ORDER_STATUS"
            elif "cancel" in normalized_message:
                intent = "ORDER_CANCEL"
            elif "modify" in normalized_message:
                intent = "ORDER_MODIFY"
            elif any(
                term in normalized_message
                for term in ("timing", "located", "location", "payment")
            ):
                intent = "RESTAURANT_INFO"
            elif "menu" in normalized_message and "want" not in normalized_message:
                intent = "MENU_QUERY"
            else:
                intent = "ORDER_CREATE"
            return FakeLLMResponse(intent)

        if "order extraction assistant" in normalized_prompt:
            message = self._customer_message(prompt, "\n\nRules:")
            return FakeLLMResponse(json.dumps(self._extract_order_items(message)))

        if "order status assistant" in normalized_prompt:
            message = self._customer_message(prompt, "\n\nIf an order ID is present")
            order_id = self._extract_order_id(message)
            return FakeLLMResponse(order_id or "UNKNOWN")

        if "order cancellation assistant" in normalized_prompt:
            message = self._customer_message(prompt, "\n\nIf an order ID is present")
            order_id = self._extract_order_id(message)
            return FakeLLMResponse(order_id or "UNKNOWN")

        if "order modification assistant" in normalized_prompt:
            message = self._customer_message(prompt, "\n\nRules:")
            normalized_message = message.casefold()
            order_id = self._extract_order_id(message) or "UNKNOWN"
            quantity_match = re.search(r"\b(\d+)\b", normalized_message)
            quantity = quantity_match.group(1) if quantity_match else "UNKNOWN"
            item_name = "Paneer Burger" if "paneer" in normalized_message else "UNKNOWN"
            return FakeLLMResponse(
                f"ORDER_ID: {order_id}\nITEM: {item_name}\nQUANTITY: {quantity}"
            )

        if "extract the menu category" in normalized_prompt:
            return FakeLLMResponse("burger")

        if "actual menu data" in normalized_prompt:
            return FakeLLMResponse("Cheese Veg Burger is on the menu.")

        if "restaurant information assistant" in normalized_prompt:
            question = prompt.split("Customer question:", 1)[1].split(
                "\n\nRestaurant information:", 1
            )[0]
            normalized_question = question.casefold()
            if "timing" in normalized_question:
                return FakeLLMResponse(
                    f"Monday hours are {RESTAURANT_INFO['timings']['monday']}."
                )
            if "located" in normalized_question or "location" in normalized_question:
                return FakeLLMResponse(RESTAURANT_INFO["address"])
            if "payment" in normalized_question:
                methods = ", ".join(RESTAURANT_INFO["payment_methods"])
                return FakeLLMResponse(f"We accept {methods}.")
            return FakeLLMResponse("That information is currently unavailable.")

        raise AssertionError(f"Unhandled LLM prompt: {prompt[:80]}")

    @staticmethod
    def _extract_order_id(message: str) -> str:
        match = re.search(r"\b[0-9a-f]{24}\b", message.casefold())
        return match.group(0) if match else ""


class MalformedOrderLLM(DeterministicLLM):
    def invoke(self, prompt: str) -> FakeLLMResponse:
        if "order extraction assistant" in prompt.casefold():
            return FakeLLMResponse("not valid JSON")
        return super().invoke(prompt)


class PartialItemModificationLLM(DeterministicLLM):
    def invoke(self, prompt: str) -> FakeLLMResponse:
        if "order modification assistant" in prompt.casefold():
            return FakeLLMResponse("ITEM: Paneer\nQUANTITY: 2")
        return super().invoke(prompt)


# Import the application with dummy credentials and a Mongo client stub. This
# ensures module initialization cannot contact Groq or MongoDB during tests.
with patch.dict(
    os.environ,
    {
        "GROQ_API_KEY": "unit-test-key",
        "MONGODB_URI": "mongodb://127.0.0.1:27017",
        "MONGODB_DATABASE": "restaurant_agent_test",
    },
):
    with patch.object(pymongo, "MongoClient", return_value=MagicMock()):
        with patch.object(langchain_groq, "ChatGroq", return_value=MagicMock()):
            from app.graph import nodes
            import app.main as app_main
            from app.main import run_agent
            from app.tools import orders as order_tools


class RestaurantAgentRegressionTests(unittest.TestCase):
    def setUp(self):
        self.orders = {}
        self.order_number = 0

        self.llm_patch = patch.object(nodes, "llm", DeterministicLLM())
        self.llm_patch.start()
        self.addCleanup(self.llm_patch.stop)

        self.create_patch = patch.object(nodes, "create_order", side_effect=self.create_order)
        self.create_mock = self.create_patch.start()
        self.addCleanup(self.create_patch.stop)

        self.get_patch = patch.object(nodes, "get_order", side_effect=self.get_order)
        self.get_mock = self.get_patch.start()
        self.addCleanup(self.get_patch.stop)

        self.cancel_patch = patch.object(nodes, "cancel_order", side_effect=self.cancel_order)
        self.cancel_mock = self.cancel_patch.start()
        self.addCleanup(self.cancel_patch.stop)

        self.modify_patch = patch.object(nodes, "modify_order", side_effect=self.modify_order)
        self.modify_mock = self.modify_patch.start()
        self.addCleanup(self.modify_patch.stop)

    def initial_state(self, order_id: str = "") -> dict:
        return {
            "user_message": "",
            "scope": "",
            "intent": "",
            "menu_query": "",
            "current_order": None,
            "order_id": order_id,
            "pending_order_confirmation": False,
            "pending_cancellation_confirmation": False,
            "response": "",
        }

    def send(self, message: str, state: dict | None = None) -> dict:
        return run_agent(message, state or self.initial_state())

    def create_order(self, items: list, total: int | float) -> str:
        self.order_number += 1
        order_id = f"{self.order_number:024x}"
        self.orders[order_id] = {
            "items": copy.deepcopy(items),
            "total": total,
            "status": "pending",
        }
        return order_id

    def get_order(self, order_id: str) -> dict | None:
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order or order["status"] != "pending":
            return False
        order["status"] = "cancelled"
        return True

    def modify_order(self, order_id: str, items: list, total: float) -> bool:
        order = self.orders.get(order_id)
        if not order or order["status"] != "pending":
            return False
        order["items"] = copy.deepcopy(items)
        order["total"] = total
        return True

    def make_order(self, items: list, total: int | float, status: str = "pending") -> str:
        order_id = self.create_order(items, total)
        self.orders[order_id]["status"] = status
        return order_id

    def assert_no_mutations(self):
        self.create_mock.assert_not_called()
        self.cancel_mock.assert_not_called()
        self.modify_mock.assert_not_called()

    def test_order_creation_stages_canonical_single_item(self):
        state = self.send("I want 2 Cheese Veg Burgers")
        menu_item = next(item for item in get_menu() if item["name"] == "Cheese Veg Burger")

        self.assertEqual(state["current_order"]["items"], [{
            "item_id": menu_item["id"],
            "name": menu_item["name"],
            "quantity": 2,
            "price": menu_item["price"],
        }])
        self.assertEqual(state["current_order"]["total"], 358)
        self.assertTrue(state["pending_order_confirmation"])
        self.assertIsNone(state["order_id"] or None)
        self.assertIn("Would you like me to place this order?", state["response"])
        self.assertEqual(self.create_mock.call_count, 0)

    def test_order_confirmation_yes_persists_exactly_once(self):
        state = self.send("I want 2 Cheese Veg Burgers")
        state = self.send("yes", state)

        self.create_mock.assert_called_once()
        self.assertEqual(len(self.orders[state["order_id"]]["items"]), 1)
        self.assertEqual(self.orders[state["order_id"]]["items"][0]["quantity"], 2)
        self.assertEqual(self.orders[state["order_id"]]["total"], 358)
        self.assertFalse(state["pending_order_confirmation"])
        self.assertIn("placed successfully", state["response"])

    def test_order_confirmation_no_discards_staged_order(self):
        state = self.send("I want 2 Cheese Veg Burgers")
        state = self.send("no", state)

        self.create_mock.assert_not_called()
        self.assertFalse(state["pending_order_confirmation"])
        self.assertIsNone(state["current_order"])
        self.assertIn("no problem", state["response"].casefold())

    def test_multi_item_order_uses_menu_prices_and_ignores_llm_price(self):
        state = self.send("I want 2 Cheese Veg Burgers and 1 Paneer Burger")
        items = state["current_order"]["items"]
        menu_by_name = {item["name"]: item for item in get_menu()}

        self.assertEqual([item["name"] for item in items], [
            "Cheese Veg Burger",
            "Paneer Burger",
        ])
        self.assertEqual([item["quantity"] for item in items], [2, 1])
        self.assertEqual(items[0]["item_id"], menu_by_name["Cheese Veg Burger"]["id"])
        self.assertEqual(items[0]["price"], menu_by_name["Cheese Veg Burger"]["price"])
        self.assertEqual(state["current_order"]["total"], 557)
        self.assertTrue(state["pending_order_confirmation"])
        self.create_mock.assert_not_called()

    def test_invalid_multi_item_order_does_not_stage_partial_order(self):
        state = self.send("I want 1 Dragon Bowl and 1 Cheese Veg Burger")

        self.assertIsNone(state["current_order"])
        self.assertFalse(state["pending_order_confirmation"])
        self.assertIn("Dragon Bowl", state["response"])
        self.assertIn("not created", state["response"])
        self.assert_no_mutations()

    def test_status_uses_previous_order_and_displays_all_items(self):
        order_id = self.make_order(
            [
                {"item_id": "burger_002", "name": "Cheese Veg Burger", "quantity": 2, "price": 179},
                {"item_id": "burger_003", "name": "Paneer Burger", "quantity": 1, "price": 199},
            ],
            557,
        )
        state = self.send("What is the status of my order?", self.initial_state(order_id))

        self.get_mock.assert_called_once_with(order_id)
        self.assertIn("Status: pending", state["response"])
        self.assertIn("Cheese Veg Burger", state["response"])
        self.assertIn("Paneer Burger", state["response"])
        self.assertIn("Total: ₹557", state["response"])

    def test_status_explicit_order_id_takes_precedence(self):
        self.make_order([], 0)
        explicit_id = "e" * 24
        self.orders[explicit_id] = {
            "items": [{"name": "Paneer Burger", "quantity": 1}],
            "total": 199,
            "status": "ready",
        }

        state = self.send(
            f"What is the status of my order {explicit_id}?",
            self.initial_state("order-1"),
        )

        self.get_mock.assert_called_once_with(explicit_id)
        self.assertIn("Status: ready", state["response"])

    def test_modification_uses_previous_order_and_canonical_price(self):
        order_id = self.make_order(
            [{"item_id": "burger_002", "name": "Cheese Veg Burger", "quantity": 1, "price": 179}],
            179,
        )
        state = self.send("modify my order to 2 Paneer Burgers", self.initial_state(order_id))
        paneer = next(item for item in get_menu() if item["name"] == "Paneer Burger")

        self.modify_mock.assert_called_once()
        self.assertEqual(self.modify_mock.call_args.args[0], order_id)
        self.assertEqual(self.orders[order_id]["items"], [{
            "item_id": paneer["id"],
            "name": paneer["name"],
            "quantity": 2,
            "price": paneer["price"],
        }])
        self.assertEqual(self.orders[order_id]["total"], 398)
        self.assertIn("modified successfully", state["response"])

    def test_cancel_no_preserves_order_and_does_not_call_cancel(self):
        order_id = self.make_order([], 0)
        state = self.send("cancel my order", self.initial_state(order_id))
        self.assertIn("Are you sure", state["response"])
        self.assertTrue(state["pending_cancellation_confirmation"])

        state = self.send("no", state)

        self.cancel_mock.assert_not_called()
        self.assertEqual(self.orders[order_id]["status"], "pending")
        self.assertFalse(state["pending_cancellation_confirmation"])
        self.assertIn("not been cancelled", state["response"])

    def test_cancel_yes_calls_cancel_once_after_confirmation(self):
        order_id = self.make_order([], 0)
        state = self.send("cancel my order", self.initial_state(order_id))
        self.cancel_mock.assert_not_called()

        state = self.send("yes", state)

        self.cancel_mock.assert_called_once_with(order_id)
        self.assertEqual(self.orders[order_id]["status"], "cancelled")
        self.assertFalse(state["pending_cancellation_confirmation"])
        self.assertIn("cancelled successfully", state["response"])

    def test_cancel_already_cancelled_order_does_not_mutate(self):
        order_id = self.make_order([], 0, status="cancelled")
        state = self.send("cancel my order", self.initial_state(order_id))

        self.cancel_mock.assert_not_called()
        self.assertIn("already been cancelled", state["response"])

    def test_cancel_missing_order_does_not_mutate(self):
        state = self.send(f"cancel order {'f' * 24}", self.initial_state())

        self.assertIn("couldn't find an order", state["response"])
        self.assertFalse(state["pending_cancellation_confirmation"])
        self.assert_no_mutations()

    def test_modify_cancelled_order_is_rejected(self):
        order_id = self.make_order([], 0, status="cancelled")
        state = self.send("modify my order to 2 Paneer Burgers", self.initial_state(order_id))

        self.modify_mock.assert_not_called()
        self.assertIn("cannot be modified", state["response"])

    def test_partial_modification_item_name_is_not_matched(self):
        order_id = self.make_order([], 0)
        nodes.llm = PartialItemModificationLLM()

        state = self.send("modify my order to 2 Paneer Burgers", self.initial_state(order_id))

        self.modify_mock.assert_not_called()
        self.assertIn("couldn't find", state["response"])

    def test_modify_missing_order_does_not_mutate(self):
        state = self.send(
            f"modify order {'f' * 24} to 2 Paneer Burgers",
            self.initial_state(),
        )

        self.assertIn("couldn't find an order", state["response"])
        self.assert_no_mutations()

    def test_status_missing_order_returns_not_found(self):
        state = self.send(f"What is the status of order {'f' * 24}?", self.initial_state())

        self.assertIn("couldn't find an order", state["response"])
        self.assert_no_mutations()

    def test_invalid_quantities_are_rejected_without_mutation(self):
        for quantity in (0, -1):
            with self.subTest(quantity=quantity):
                state = self.send(f"I want {quantity} Cheese Veg Burgers")
                self.assertIsNone(state["current_order"])
                self.assertFalse(state["pending_order_confirmation"])
                self.assertIn("valid quantity", state["response"])
                self.assert_no_mutations()

    def test_unavailable_item_rejects_order(self):
        menu = copy.deepcopy(get_menu())
        next(item for item in menu if item["name"] == "Cheese Veg Burger")["available"] = False
        menu_patch = patch.object(nodes, "get_menu", return_value=menu)
        menu_patch.start()
        self.addCleanup(menu_patch.stop)

        state = self.send("I want 2 Cheese Veg Burgers")

        self.assertIsNone(state["current_order"])
        self.assertFalse(state["pending_order_confirmation"])
        self.assertIn("currently unavailable", state["response"])
        self.assert_no_mutations()

    def test_unknown_item_rejects_order_without_partial_creation(self):
        state = self.send("I want 1 Dragon Bowl and 1 Cheese Veg Burger")

        self.assertIsNone(state["current_order"])
        self.assertFalse(state["pending_order_confirmation"])
        self.assert_no_mutations()

    def test_pending_confirmation_unknown_reply_keeps_order_staged(self):
        state = self.send("I want 2 Cheese Veg Burgers")
        state = self.send("maybe", state)

        self.create_mock.assert_not_called()
        self.assertTrue(state["pending_order_confirmation"])
        self.assertIsNotNone(state["current_order"])
        self.assertIn("replying 'yes' or 'no'", state["response"])

    def test_restaurant_timings_use_configured_data_without_db_calls(self):
        state = self.send("What are your timings?")

        self.assertIn(RESTAURANT_INFO["timings"]["monday"], state["response"])
        self.assertNotIn("24/7", state["response"])
        self.assert_no_mutations()
        self.get_mock.assert_not_called()

    def test_restaurant_location_uses_configured_data_without_db_calls(self):
        state = self.send("Where are you located?")

        self.assertEqual(state["response"], RESTAURANT_INFO["address"])
        self.assert_no_mutations()
        self.get_mock.assert_not_called()

    def test_payment_methods_use_configured_data_without_db_calls(self):
        state = self.send("What payment methods do you accept?")

        for method in RESTAURANT_INFO["payment_methods"]:
            self.assertIn(method, state["response"])
        self.assert_no_mutations()
        self.get_mock.assert_not_called()

    def test_out_of_scope_request_does_not_touch_orders(self):
        state = self.send("Explain binary search in Java")

        self.assertIn("only help with restaurant-related", state["response"])
        self.assert_no_mutations()
        self.get_mock.assert_not_called()

    def test_menu_query_still_routes_through_menu_node(self):
        state = self.send("Show me burgers on the menu")

        self.assertEqual(state["menu_query"], "burger")
        self.assertIn("Cheese Veg Burger", state["response"])
        self.assert_no_mutations()

    def test_mongodb_ping_helper_issues_only_ping_command(self):
        command_patch = patch.object(order_tools.db, "command", return_value={"ok": 1})
        command_mock = command_patch.start()
        self.addCleanup(command_patch.stop)

        self.assertTrue(order_tools.ping_mongodb())
        command_mock.assert_called_once_with("ping")

    def test_order_timestamps_are_timezone_aware_utc(self):
        def assert_utc_timestamp(timestamp):
            self.assertIsInstance(timestamp, datetime)
            self.assertEqual(timestamp.tzinfo, UTC)
            self.assertEqual(timestamp.utcoffset(), timedelta(0))

        inserted_id = ObjectId()
        with patch.object(
            order_tools.orders_collection,
            "insert_one",
            return_value=MagicMock(inserted_id=inserted_id),
        ) as insert_one:
            order_tools.create_order([], 0)

        created_document = insert_one.call_args.args[0]
        self.assertEqual(set(created_document), {
            "items", "total", "status", "created_at",
        })
        assert_utc_timestamp(created_document["created_at"])

        order_id = str(ObjectId())
        with patch.object(
            order_tools.orders_collection,
            "update_one",
            return_value=MagicMock(modified_count=1),
        ) as update_one:
            self.assertTrue(order_tools.cancel_order(order_id))

        cancel_update = update_one.call_args.args[1]["$set"]
        self.assertEqual(set(cancel_update), {"status", "updated_at"})
        assert_utc_timestamp(cancel_update["updated_at"])

        with patch.object(
            order_tools.orders_collection,
            "update_one",
            return_value=MagicMock(modified_count=1),
        ) as update_one:
            self.assertTrue(order_tools.modify_order(order_id, [], 0))

        modify_update = update_one.call_args.args[1]["$set"]
        self.assertEqual(set(modify_update), {"items", "total", "updated_at"})
        assert_utc_timestamp(modify_update["updated_at"])

    def test_order_timestamps_preserve_utc_instants_through_bson_round_trip(self):
        created_at = datetime.now(UTC).replace(microsecond=123000)
        updated_at = datetime.now(UTC).replace(microsecond=456000)
        order_document = {
            "items": [],
            "total": 0,
            "status": "pending",
            "created_at": created_at,
            "updated_at": updated_at,
        }

        encoded = BSON.encode(order_document)

        # PyMongo's default BSON decoder returns naive datetimes representing UTC.
        default_decoded = BSON(encoded).decode()
        for field, original in (
            ("created_at", created_at),
            ("updated_at", updated_at),
        ):
            decoded = default_decoded[field]
            self.assertIsNone(decoded.tzinfo)
            self.assertEqual(decoded.replace(tzinfo=UTC), original)

        utc_codec = CodecOptions(tz_aware=True, tzinfo=UTC)
        aware_decoded = BSON(encoded).decode(codec_options=utc_codec)
        for field, original in (
            ("created_at", created_at),
            ("updated_at", updated_at),
        ):
            decoded = aware_decoded[field]
            self.assertEqual(decoded.tzinfo, UTC)
            self.assertEqual(decoded.utcoffset(), timedelta(0))
            self.assertEqual(decoded, original)

    def test_llm_failure_returns_safe_response_and_logs_no_error_text(self):
        secret_marker = "do-not-log-private-error"
        nodes.llm = MagicMock()
        nodes.llm.invoke.side_effect = RuntimeError(secret_marker)

        with self.assertLogs("app.graph.nodes", level="ERROR") as captured:
            state = self.send("What are your timings?")

        self.assertIn("trouble processing your request", state["response"])
        self.assertNotIn(secret_marker, "\n".join(captured.output))

    def test_missing_llm_content_returns_safe_response(self):
        nodes.llm = MagicMock()
        nodes.llm.invoke.return_value = object()

        with self.assertLogs("app.graph.nodes", level="ERROR"):
            state = self.send("What are your timings?")

        self.assertIn("trouble processing your request", state["response"])

    def test_unavailable_llm_client_returns_safe_response(self):
        nodes.llm = None

        with self.assertLogs("app.graph.nodes", level="ERROR"):
            state = self.send("What are your timings?")

        self.assertIn("trouble processing your request", state["response"])

    def test_malformed_order_json_is_rejected_without_creation(self):
        nodes.llm = MalformedOrderLLM()

        state = self.send("I want 2 Cheese Veg Burgers")

        self.assertIn("couldn't identify all the requested items", state["response"])
        self.assertIsNone(state["current_order"])
        self.assertFalse(state["pending_order_confirmation"])
        self.create_mock.assert_not_called()

    def test_lookup_failure_returns_order_system_response(self):
        order_id = self.make_order([], 0)
        self.get_mock.side_effect = order_tools.OrderStoreError("private detail")

        with self.assertLogs("app.main", level="ERROR") as captured:
            state = self.send("What is the status of my order?", self.initial_state(order_id))

        self.assertIn("trouble accessing the order system", state["response"])
        self.assertNotIn("private detail", "\n".join(captured.output))

    def test_create_failure_does_not_report_success_and_keeps_confirmation(self):
        state = self.send("I want 2 Cheese Veg Burgers")
        self.create_mock.side_effect = order_tools.OrderStoreError("private detail")

        with self.assertLogs("app.main", level="ERROR"):
            state = self.send("yes", state)

        self.assertIn("trouble accessing the order system", state["response"])
        self.assertNotIn("successfully", state["response"])
        self.assertTrue(state["pending_order_confirmation"])
        self.assertIsNotNone(state["current_order"])

    def test_modify_failure_returns_safe_order_system_response(self):
        order_id = self.make_order([], 0)
        self.modify_mock.side_effect = order_tools.OrderStoreError("private detail")

        with self.assertLogs("app.main", level="ERROR"):
            state = self.send("modify my order to 2 Paneer Burgers", self.initial_state(order_id))

        self.assertIn("trouble accessing the order system", state["response"])
        self.assertNotIn("successfully", state["response"])

    def test_cancel_failure_returns_safe_order_system_response_and_keeps_pending(self):
        order_id = self.make_order([], 0)
        state = self.send("cancel my order", self.initial_state(order_id))
        self.cancel_mock.side_effect = order_tools.OrderStoreError("private detail")

        with self.assertLogs("app.main", level="ERROR"):
            state = self.send("yes", state)

        self.assertIn("trouble accessing the order system", state["response"])
        self.assertNotIn("successfully", state["response"])
        self.assertTrue(state["pending_cancellation_confirmation"])

    def test_mongodb_helpers_translate_driver_failures(self):
        order_id = "a" * 24
        operations = (
            ("insert_one", lambda: order_tools.create_order([], 0)),
            ("find_one", lambda: order_tools.get_order(order_id)),
            ("update_one", lambda: order_tools.cancel_order(order_id)),
            ("update_one", lambda: order_tools.modify_order(order_id, [], 0)),
        )

        for operation_name, operation in operations:
            with self.subTest(operation=operation_name, call=operation):
                with patch.object(
                    order_tools.orders_collection,
                    operation_name,
                    side_effect=PyMongoError("private database detail"),
                ):
                    with self.assertLogs("app.tools.orders", level="ERROR"):
                        with self.assertRaises(order_tools.OrderStoreError):
                            operation()

    def test_missing_menu_data_returns_safe_response(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            menu_path = Path(temporary_directory) / "missing-menu.json"
            menu_patch = patch("app.tools.menu.MENU_FILE", menu_path)
            menu_patch.start()
            self.addCleanup(menu_patch.stop)

            with self.assertLogs(level="ERROR"):
                state = self.send("I want 2 Cheese Veg Burgers")

        self.assertIn("menu is temporarily unavailable", state["response"])
        self.assertIsNone(state["current_order"])
        self.create_mock.assert_not_called()

    def test_malformed_menu_structure_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            menu_path = Path(temporary_directory) / "menu.json"
            menu_path.write_text('{"items": [{}]}', encoding="utf-8")
            with patch("app.tools.menu.MENU_FILE", menu_path):
                with self.assertLogs("app.tools.menu", level="ERROR"):
                    with self.assertRaises(MenuDataError):
                        get_menu()

    def test_empty_input_does_not_call_llm(self):
        nodes.llm = MagicMock()

        state = self.send(" \t ")

        self.assertIn("enter a message", state["response"])
        nodes.llm.invoke.assert_not_called()

    def test_invalid_explicit_order_id_does_not_fall_back_to_previous_order(self):
        previous_order_id = self.make_order([], 0)

        state = self.send(
            "What is the status of order invalid-id?",
            self.initial_state(previous_order_id),
        )

        self.get_mock.assert_not_called()
        self.assertIn("provide your order ID", state["response"])

    def test_create_without_returned_order_id_does_not_report_success(self):
        state = self.send("I want 2 Cheese Veg Burgers")
        self.create_mock.side_effect = None
        self.create_mock.return_value = None

        with self.assertLogs("app.main", level="ERROR"):
            state = self.send("yes", state)

        self.assertIn("trouble accessing the order system", state["response"])
        self.assertNotIn("successfully", state["response"])
        self.assertTrue(state["pending_order_confirmation"])

    def test_unexpected_graph_exception_returns_safe_response(self):
        graph = app_main.graph
        with patch.object(graph, "invoke", side_effect=RuntimeError("private failure detail")):
            with self.assertLogs("app.main", level="ERROR") as captured:
                state = self.send("What are your timings?")

        self.assertIn("something went wrong", state["response"])
        self.assertNotIn("private failure detail", "\n".join(captured.output))

    def test_keyboard_interrupt_is_not_swallowed(self):
        graph = app_main.graph
        with patch.object(graph, "invoke", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.send("What are your timings?")

    def test_cli_handles_empty_input_and_normal_exit(self):
        nodes.llm = MagicMock()
        output = io.StringIO()

        with patch("builtins.input", side_effect=["  ", "exit"]):
            with redirect_stdout(output):
                runpy.run_path(app_main.__file__, run_name="__main__")

        self.assertIn("enter a message", output.getvalue())
        self.assertIn("Goodbye", output.getvalue())
        nodes.llm.invoke.assert_not_called()

    def test_cli_handles_eof_as_normal_exit(self):
        output = io.StringIO()

        with patch("builtins.input", side_effect=EOFError):
            with redirect_stdout(output):
                runpy.run_path(app_main.__file__, run_name="__main__")

        self.assertIn("Goodbye", output.getvalue())


if __name__ == "__main__":
    unittest.main()
