import logging
import json
import math
from pathlib import Path


MENU_FILE = Path(__file__).resolve().parent.parent / "data" / "menu.json"
logger = logging.getLogger(__name__)


class MenuDataError(RuntimeError):
    """Raised when the canonical menu cannot be safely used."""


def get_menu():
    """Return all menu items."""
    try:
        with open(MENU_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        logger.error("Menu data could not be loaded (%s).", type(error).__name__)
        raise MenuDataError("Menu data is unavailable.") from None

    if not isinstance(data, dict):
        logger.error("Menu data has an invalid top-level structure.")
        raise MenuDataError("Menu data is invalid.")

    items = data.get("items")
    if not isinstance(items, list) or not items:
        logger.error("Menu data does not contain a valid item list.")
        raise MenuDataError("Menu data is invalid.")

    item_ids = set()
    required_fields = {"id", "name", "category", "description", "price", "available"}
    for item in items:
        if not isinstance(item, dict) or not required_fields.issubset(item):
            logger.error("Menu data contains an incomplete item record.")
            raise MenuDataError("Menu data is invalid.")

        if any(
            not isinstance(item[field], str) or not item[field].strip()
            for field in ("id", "name", "category")
        ) or not isinstance(item["description"], str):
            logger.error("Menu data contains invalid item fields.")
            raise MenuDataError("Menu data is invalid.")

        price = item["price"]
        try:
            valid_price = (
                isinstance(price, (int, float))
                and not isinstance(price, bool)
                and math.isfinite(price)
                and price >= 0
            )
        except (OverflowError, TypeError):
            valid_price = False

        if not valid_price or not isinstance(item["available"], bool):
            logger.error("Menu data contains invalid price or availability.")
            raise MenuDataError("Menu data is invalid.")

        if item["id"] in item_ids:
            logger.error("Menu data contains duplicate item identifiers.")
            raise MenuDataError("Menu data is invalid.")
        item_ids.add(item["id"])

    return items


def get_available_menu():
    """Return only currently available menu items."""
    return [
        item
        for item in get_menu()
        if item["available"]
    ]


def get_menu_item(item_id: str):
    """Find a menu item by its ID."""
    for item in get_menu():
        if item["id"] == item_id:
            return item

    return None


def search_menu(query: str):
    """Search menu items by name or category."""
    if not isinstance(query, str):
        raise ValueError("Menu query must be text.")

    query = query.lower().strip()

    results = []

    for item in get_available_menu():
        if (
            query in item["name"].lower()
            or query in item["category"].lower()
        ):
            results.append(item)

    return results


def check_item_exists(item_name: str):
    """Check whether a menu item exists."""
    item_name = item_name.lower().strip()

    for item in get_menu():
        if item["name"].lower() == item_name:
            return item

    return None

if __name__ == "__main__":
    print("ALL MENU:")
    print(get_menu())

    print("\nBURGERS:")
    print(search_menu("burger"))

    print("\nCHEESE VEG BURGER:")
    print(check_item_exists("Cheese Veg Burger"))