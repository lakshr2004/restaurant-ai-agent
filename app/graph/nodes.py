import logging
import json
import re

from app.tools.orders import (
    create_order,
    get_order,
    cancel_order,
    modify_order,
    OrderStoreError,
)

from app.data.restaurant_info import RESTAURANT_INFO

from app.graph.state import OrderItem, RestaurantState

from app.agents.restaurant_agent import get_llm

from app.tools.menu import MenuDataError, get_menu, search_menu


logger = logging.getLogger(__name__)

try:
    llm = get_llm()
except Exception as error:
    logger.error("LLM client initialization failed (%s).", type(error).__name__)
    llm = None


class LLMServiceError(RuntimeError):
    """Raised when the LLM cannot provide usable text."""


def _invoke_llm(prompt: str, operation: str) -> str:
    try:
        if llm is None:
            raise RuntimeError("LLM client is unavailable.")
        result = llm.invoke(prompt)
        content = getattr(result, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text.")
    except Exception as error:
        logger.error("LLM %s failed (%s).", operation, type(error).__name__)
        raise LLMServiceError("LLM service is temporarily unavailable.") from None

    return content.strip()


def _resolve_order_id(user_message: str, previous_order_id: str) -> str:
    explicit_ids = re.findall(r"\b[0-9a-fA-F]{24}\b", user_message)
    if len(explicit_ids) == 1:
        return explicit_ids[0]
    if len(explicit_ids) > 1:
        return ""

    explicit_reference = re.search(
        r"\border(?:\s+id)?\s+([a-zA-Z0-9_-]+)\b",
        user_message,
        re.IGNORECASE,
    )
    if explicit_reference and explicit_reference.group(1).casefold() not in {
        "to",
        "now",
        "please",
        "today",
        "right",
    }:
        return ""

    return previous_order_id


# ============================================================
# SCOPE GUARD
# ============================================================

def scope_guard(state: RestaurantState) -> RestaurantState:

    user_message = state["user_message"]

    prompt = f"""
You are a restaurant request classifier.

Classify the customer's message into exactly one of these categories:

RESTAURANT

- Menu questions
- Food or dish requests
- Ordering food
- Quantity changes
- Order modification
- Order cancellation
- Order status
- Restaurant timings
- Restaurant policies
- Prices
- Ingredients or allergens
- Any other restaurant-related request

OUT_OF_SCOPE

- Programming
- General knowledge
- Homework
- Politics
- News
- Entertainment
- Personal advice
- Any request unrelated to this restaurant

Customer message:

{user_message}

Reply with ONLY:

RESTAURANT

or

OUT_OF_SCOPE
"""

    classification = _invoke_llm(prompt, "scope classification").upper()

    if classification not in {
        "RESTAURANT",
        "OUT_OF_SCOPE",
    }:
        classification = "OUT_OF_SCOPE"

    return {
        **state,
        "scope": classification,
    }


# ============================================================
# INTENT ROUTER
# ============================================================

def intent_router(state: RestaurantState) -> RestaurantState:

    user_message = state["user_message"]

    prompt = f"""
You are an intent classifier for a restaurant AI agent.

Classify the customer message into exactly ONE category:

MENU_QUERY

ORDER_CREATE

ORDER_MODIFY

ORDER_CANCEL

ORDER_STATUS

RESTAURANT_INFO

OUT_OF_SCOPE

Definitions:

MENU_QUERY:
Customer wants to see or ask about menu items.

ORDER_CREATE:
Customer wants to order food.

ORDER_MODIFY:
Customer wants to change an existing order.

ORDER_CANCEL:
Customer wants to cancel an existing order.

ORDER_STATUS:
Customer asks about the status of an existing order.

RESTAURANT_INFO:
Customer asks about restaurant timings, policies, location,
ingredients, allergens, pricing, etc.

OUT_OF_SCOPE:
Anything unrelated to the restaurant.

Customer message:

{user_message}

Reply with ONLY the category name.
"""

    intent = _invoke_llm(prompt, "intent classification").upper()

    valid_intents = {
        "MENU_QUERY",
        "ORDER_CREATE",
        "ORDER_MODIFY",
        "ORDER_CANCEL",
        "ORDER_STATUS",
        "RESTAURANT_INFO",
        "OUT_OF_SCOPE",
    }

    if intent not in valid_intents:
        intent = "OUT_OF_SCOPE"

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# MENU NODE
# ============================================================

def menu_node(state: RestaurantState) -> RestaurantState:

    user_message = state["user_message"]

    prompt = f"""
You are a restaurant menu assistant.

Extract the menu category from the customer's message.

Possible categories:

burger
pizza
pasta
fries
coffee
beverages
menu

Customer message:

{user_message}

Reply with ONLY the category.
"""

    query = _invoke_llm(prompt, "menu query extraction").lower()

    valid_categories = {
        "burger",
        "pizza",
        "pasta",
        "fries",
        "coffee",
        "beverages",
        "menu",
    }

    if query not in valid_categories:
        query = "menu"

    # --------------------------------------------------------
    # Generic menu question
    # --------------------------------------------------------

    if query == "menu":

        return {
            **state,
            "menu_query": query,
            "response": (
                "Sure! I can help you explore our menu. "
                "What would you like to know about?"
            ),
        }

    # --------------------------------------------------------
    # Search menu
    # --------------------------------------------------------

    menu_items = search_menu(query)

    if not menu_items:

        return {
            **state,
            "menu_query": query,
            "response": (
                "Sorry, I couldn't find matching items "
                "on our menu."
            ),
        }

    # --------------------------------------------------------
    # Prepare menu data
    # --------------------------------------------------------

    menu_text = "\n".join(
        f"- {item['name']} - ₹{item['price']}: "
        f"{item['description']}"
        for item in menu_items
    )

    prompt = f"""
You are a restaurant assistant.

Customer asked:

{user_message}

Actual menu data:

{menu_text}

Answer using ONLY the menu data above.

Do not invent dishes, prices, ingredients,
or availability.

Keep the response concise and friendly.
"""

    response = _invoke_llm(prompt, "menu response")

    return {
        **state,
        "menu_query": query,
        "response": response,
    }


# ============================================================
# ORDER CREATE NODE
# ============================================================

def _parse_order_items(extracted: str) -> list[tuple[str, int]] | None:
    extracted = extracted.strip()

    if extracted.startswith("```"):
        lines = extracted.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        extracted = "\n".join(lines).strip()

    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    extracted_items = payload.get("items")
    if not isinstance(extracted_items, list) or not extracted_items:
        return None

    requested_items = []
    for entry in extracted_items:
        if not isinstance(entry, dict):
            return None

        item_name = entry.get("name")
        if not isinstance(item_name, str) or not item_name.strip():
            return None

        quantity_value = entry.get("quantity", 1)
        if isinstance(quantity_value, bool):
            return None

        try:
            quantity = int(quantity_value)
        except (TypeError, ValueError):
            return None

        if isinstance(quantity_value, float) and not quantity_value.is_integer():
            return None

        requested_items.append((item_name.strip(), quantity))

    return requested_items


def order_node(state: RestaurantState) -> RestaurantState:
    user_message = state["user_message"]
    menu_items = get_menu()
    menu_names = "\n".join(f"- {item['name']}" for item in menu_items)

    prompt = f"""
You are an order extraction assistant for a restaurant.

Extract every distinct food item and its quantity from the customer's
message. Available canonical menu item names:

{menu_names}

Customer message:
{user_message}

Rules:
- Use the exact canonical menu name when the request matches a menu item.
- Map common names such as "fries" to the matching canonical menu item.
- If quantity is omitted, use 1.
- Preserve an unmatched requested name so it can be checked against the menu.
- Do not omit requested items, and do not return prices, IDs, availability,
  or a total.

Return only valid JSON in this format:
{{"items": [{{"name": "Cheese Veg Burger", "quantity": 2}}]}}
"""

    extracted = _invoke_llm(prompt, "order extraction")
    requested_items = _parse_order_items(extracted)

    if requested_items is None:
        return {
            **state,
            "response": (
                "I couldn't identify all the requested items. "
                "Please list the items again. I have not created the order."
            ),
        }

    order_items: list[OrderItem] = []
    total = 0

    for requested_name, quantity in requested_items:
        selected_item = next(
            (
                item
                for item in menu_items
                if item["name"].strip().casefold()
                == requested_name.casefold()
            ),
            None,
        )

        if selected_item is None:
            return {
                **state,
                "response": (
                    f"Sorry, {requested_name} is not on our menu. "
                    "I have not created the order."
                ),
            }

        if quantity <= 0:
            return {
                **state,
                "response": (
                    f"Please provide a valid quantity for "
                    f"{selected_item['name']}. I have not created the order."
                ),
            }

        if not selected_item.get("available", False):
            return {
                **state,
                "response": (
                    f"Sorry, {selected_item['name']} is currently unavailable. "
                    "I have not created the order."
                ),
            }

        price = selected_item["price"]
        order_items.append(
            {
                "item_id": selected_item["id"],
                "name": selected_item["name"],
                "quantity": quantity,
                "price": price,
            }
        )
        total += price * quantity

    order_summary = "\n".join(
        f"- {item['quantity']} × {item['name']} — "
        f"₹{item['price'] * item['quantity']}"
        for item in order_items
    )

    return {
        **state,
        "current_order": {
            "items": order_items,
            "total": total,
        },
        "pending_order_confirmation": True,
        "response": (
            "Sure! Here's your order:\n\n"
            f"{order_summary}\n\n"
            f"Total: ₹{total}\n\n"
            "Would you like me to place this order?"
        ),
    }


# ============================================================
# CONFIRMATION NODE
# ============================================================

def confirmation_node(
    state: RestaurantState
) -> RestaurantState:

    user_message = state["user_message"].strip().lower()

    # --------------------------------------------------------
    # YES
    # --------------------------------------------------------

    if user_message in {
        "yes",
        "y",
        "yeah",
        "yep",
        "sure",
        "confirm",
        "confirmed",
        "place it",
        "place the order",
        "yes please",
        "sure please",
    }:

        current_order = state.get("current_order")

        if not current_order:

            return {
                **state,
                "pending_order_confirmation": False,
                "current_order": None,
                "response": (
                    "I couldn't find a pending order "
                    "to confirm."
                ),
            }

        order_items = current_order.get("items", [])
        order_total = current_order.get("total", 0)

        if not order_items:

            return {
                **state,
                "pending_order_confirmation": False,
                "current_order": None,
                "response": (
                    "I couldn't find a pending order "
                    "to confirm."
                ),
            }

        # ----------------------------------------------------
        # Create order only after confirmation
        # ----------------------------------------------------

        order_id = create_order(
            order_items,
            order_total,
        )

        if not isinstance(order_id, str) or not order_id.strip():
            logger.error("Order creation returned no order identifier.")
            raise OrderStoreError("Order creation was not confirmed.")

        return {
            **state,
            "pending_order_confirmation": False,
            "order_id": order_id,
            "response": (
                "Your order has been placed successfully! 🎉\n\n"
                f"Order ID: {order_id}\n"
                f"Total: ₹{order_total}"
            ),
        }

    # --------------------------------------------------------
    # NO
    # --------------------------------------------------------

    if user_message in {
        "no",
        "n",
        "nope",
        "cancel",
        "cancel it",
        "don't place it",
        "do not place it",
        "no thanks",
        "not now",
    }:

        return {
            **state,
            "pending_order_confirmation": False,
            "current_order": None,
            "response": (
                "No problem! Your order has been cancelled."
            ),
        }

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return {
        **state,
        "response": (
            "Please confirm your order by replying "
            "'yes' or 'no'."
        ),
    }


# ============================================================
# ORDER STATUS NODE
# ============================================================

def order_status_node(
    state: RestaurantState
) -> RestaurantState:

    user_message = state["user_message"].strip()
    order_id = _resolve_order_id(user_message, state.get("order_id", ""))

    if not order_id:

        return {
            **state,
            "response": (
                "Sure! Please provide your order ID "
                "so I can check its status."
            ),
        }

    # --------------------------------------------------------
    # Get order
    # --------------------------------------------------------

    order = get_order(order_id)

    if not order:

        return {
            **state,
            "response": (
                f"Sorry, I couldn't find an order "
                f"with ID {order_id}."
            ),
        }

    items = order.get("items", [])
    total = order.get("total", 0)
    status = order.get("status", "pending")

    item_text = "\n".join(
        f"- {item['quantity']} × {item['name']}"
        for item in items
    )

    return {
        **state,
        "order_id": order_id,
        "response": (
            f"Order ID: {order_id}\n\n"
            f"Status: {status}\n\n"
            f"Items:\n"
            f"{item_text}\n\n"
            f"Total: ₹{total}"
        ),
    }


# ============================================================
# ORDER CANCEL NODE
# ============================================================

def order_cancel_node(
    state: RestaurantState
) -> RestaurantState:

    user_message = state["user_message"].strip()
    order_id = _resolve_order_id(user_message, state.get("order_id", ""))

    # --------------------------------------------------------
    # No Order ID
    # --------------------------------------------------------

    if not order_id:

        return {
            **state,
            "response": (
                "Sure! Please provide your order ID "
                "so I can cancel it."
            ),
        }

    # --------------------------------------------------------
    # Find Order
    # --------------------------------------------------------

    order = get_order(order_id)

    if not order:

        return {
            **state,
            "response": (
                f"Sorry, I couldn't find an order "
                f"with ID {order_id}."
            ),
        }

    # --------------------------------------------------------
    # Check Current Status
    # --------------------------------------------------------

    status = order.get(
        "status",
        "pending"
    )

    # --------------------------------------------------------
    # Already Cancelled
    # --------------------------------------------------------

    if status == "cancelled":

        return {
            **state,
            "response": (
                f"Order {order_id} has already been cancelled."
            ),
        }

    # --------------------------------------------------------
    # Only pending orders can be cancelled
    # --------------------------------------------------------

    if status != "pending":

        return {
            **state,
            "response": (
                f"Sorry, order {order_id} cannot be cancelled "
                f"because its current status is '{status}'."
            ),
        }

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT cancel here.
    #
    # Store the order ID and ask for confirmation.
    # --------------------------------------------------------

    return {
        **state,

        "order_id": order_id,
        "pending_cancellation_confirmation": True,

        "response": (
            f"Are you sure you want to cancel "
            f"order {order_id}?\n\n"
            f"Reply 'yes' to confirm or 'no' to keep the order."
        ),
    }

# ============================================================
# CANCELLATION CONFIRMATION NODE
# ============================================================

def cancellation_confirmation_node(
    state: RestaurantState
) -> RestaurantState:

    user_message = state["user_message"].strip().lower()

    order_id = state.get("order_id", "")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not order_id:

        return {
            **state,

            "pending_cancellation_confirmation": False,

            "response": (
                "I couldn't find a pending cancellation request."
            ),
        }

    # --------------------------------------------------------
    # YES
    # --------------------------------------------------------

    if user_message in {
        "yes",
        "y",
        "yeah",
        "yep",
        "sure",
        "confirm",
        "confirmed",
        "cancel it",
        "yes please",
    }:

        cancelled = cancel_order(order_id)

        if not cancelled:

            return {
                **state,

                "pending_cancellation_confirmation": False,

                "response": (
                    "Sorry, I couldn't cancel the order. "
                    "Please try again."
                ),
            }

        return {
            **state,

            "pending_cancellation_confirmation": False,

            "response": (
                "Your order has been cancelled successfully! "
                "✅\n\n"
                f"Order ID: {order_id}"
            ),
        }

    # --------------------------------------------------------
    # NO
    # --------------------------------------------------------

    if user_message in {
        "no",
        "n",
        "nope",
        "don't cancel",
        "do not cancel",
        "keep it",
        "no thanks",
        "not now",
    }:

        return {
            **state,

            "pending_cancellation_confirmation": False,

            "response": (
                "No problem! Your order has not been cancelled. 👍"
            ),
        }

    # --------------------------------------------------------
    # UNKNOWN RESPONSE
    # --------------------------------------------------------

    return {
        **state,

        "response": (
            "Please confirm the cancellation by replying "
            "'yes' or 'no'."
        ),
    }
# ============================================================
# ORDER MODIFY NODE
# ============================================================

def order_modify_node(
    state: RestaurantState
) -> RestaurantState:

    user_message = state["user_message"].strip()

    menu_items = get_menu()
    menu_names = "\n".join(f"- {item['name']}" for item in menu_items)

    prompt = f"""
You are an order modification assistant for a restaurant.

Extract the requested food item and quantity.

Available canonical menu items:
{menu_names}

Customer message:

{user_message}

Rules:

- Use the EXACT menu item name.
- Never make item names plural.
- If food item is missing, return UNKNOWN.
- If quantity is missing, return UNKNOWN.

Return exactly:

ITEM: <exact menu item name or UNKNOWN>
QUANTITY: <number or UNKNOWN>

Do not add anything else.
"""

    extracted = _invoke_llm(prompt, "order modification extraction")

    item_name = ""
    quantity = None

    # --------------------------------------------------------
    # Parse LLM response
    # --------------------------------------------------------

    for line in extracted.splitlines():

        line = line.strip()

        if line.upper().startswith("ITEM:"):

            item_name = line.split(":", 1)[1].strip()

        elif line.upper().startswith("QUANTITY:"):

            value = line.split(":", 1)[1].strip()

            if value.upper() != "UNKNOWN":

                try:
                    quantity = int(value)
                except ValueError:
                    quantity = None

    order_id = _resolve_order_id(user_message, state.get("order_id", ""))

    if not order_id:

        return {
            **state,
            "response": (
                "Sure! Please provide your order ID "
                "so I can modify your order."
            ),
        }

    # --------------------------------------------------------
    # Validate item
    # --------------------------------------------------------

    if not item_name or item_name.upper() == "UNKNOWN":

        return {
            **state,
            "order_id": order_id,
            "response": (
                "Please tell me which item you'd like "
                "to change in your order."
            ),
        }

    # --------------------------------------------------------
    # Validate quantity
    # --------------------------------------------------------

    if quantity is None or quantity <= 0:

        return {
            **state,
            "order_id": order_id,
            "response": (
                "Please provide a valid quantity."
            ),
        }

    # --------------------------------------------------------
    # Find existing order
    # --------------------------------------------------------

    order = get_order(order_id)

    if not order:

        return {
            **state,
            "order_id": order_id,
            "response": (
                f"Sorry, I couldn't find an order "
                f"with ID {order_id}."
            ),
        }

    # --------------------------------------------------------
    # Check order status
    # --------------------------------------------------------

    status = order.get("status", "pending")

    if status != "pending":

        return {
            **state,
            "order_id": order_id,
            "response": (
                f"Sorry, order {order_id} cannot be modified "
                f"because its current status is '{status}'."
            ),
        }

    # --------------------------------------------------------
    # Search menu
    # --------------------------------------------------------

    selected_item = next(
        (
            item
            for item in menu_items
            if item["name"].strip().casefold() == item_name.strip().casefold()
        ),
        None,
    )

    if selected_item is None:

        return {
            **state,
            "order_id": order_id,
            "response": (
                f"Sorry, I couldn't find "
                f"'{item_name}' on our menu."
            ),
        }

    # --------------------------------------------------------
    # Check availability
    # --------------------------------------------------------

    if not selected_item.get("available", False):

        return {
            **state,
            "order_id": order_id,
            "response": (
                f"Sorry, {selected_item['name']} is "
                f"currently unavailable."
            ),
        }

    # --------------------------------------------------------
    # Calculate new order
    # --------------------------------------------------------

    price = selected_item["price"]

    new_total = price * quantity

    new_order_item: OrderItem = {
        "item_id": selected_item["id"],
        "name": selected_item["name"],
        "quantity": quantity,
        "price": price,
    }

    # --------------------------------------------------------
    # Modify MongoDB order
    # --------------------------------------------------------

    modified = modify_order(
        order_id,
        [new_order_item],
        new_total,
    )

    if not modified:

        return {
            **state,
            "order_id": order_id,
            "response": (
                "Sorry, I couldn't modify your order. "
                "Please try again."
            ),
        }

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    return {
        **state,
        "order_id": order_id,
        "current_order": {
            "items": [new_order_item],
            "total": new_total,
        },
        "response": (
            "Your order has been modified successfully! ✅\n\n"
            f"Order ID: {order_id}\n"
            f"Item: {quantity} × {selected_item['name']}\n"
            f"Total: ₹{new_total}"
        ),
    }


# ============================================================
# RESTAURANT INFO NODE
# ============================================================

def restaurant_info_node(
    state: RestaurantState
) -> RestaurantState:

    user_message = state["user_message"]

    restaurant_data = str(RESTAURANT_INFO)

    prompt = f"""
You are a restaurant information assistant.

Customer question:

{user_message}

Restaurant information:

{restaurant_data}

IMPORTANT RULES:

1. Answer ONLY using the restaurant information provided above.

2. Do NOT invent restaurant details.

3. If the requested information is not available,
   say that the information is currently unavailable.

4. Keep the answer concise and friendly.

5. Do not answer unrelated questions.
"""

    response = _invoke_llm(prompt, "restaurant information")

    return {
        **state,
        "response": response,
    }


# ============================================================
# OUT OF SCOPE NODE
# ============================================================

def out_of_scope_node(
    state: RestaurantState
) -> RestaurantState:

    return {
        **state,
        "response": (
            "Sorry, I can only help with restaurant-related "
            "queries and orders."
        ),
    }