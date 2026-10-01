import logging
import os
from typing import NoReturn

from datetime import UTC, datetime

from bson import ObjectId
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError, ServerSelectionTimeoutError


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
logger = logging.getLogger(__name__)


class OrderStoreError(RuntimeError):
    """Raised when an order storage operation cannot be completed."""


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = None
db = None
orders_collection = None

if MONGODB_URI and MONGODB_DATABASE:
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        orders_collection = db["orders"]
    except Exception as error:
        logger.error("MongoDB client initialization failed (%s).", type(error).__name__)
        client = None
        db = None
        orders_collection = None


def _require_orders_collection():
    if orders_collection is None:
        raise OrderStoreError("MongoDB is not configured or unavailable.")
    return orders_collection


def _raise_storage_error(operation: str, error: Exception) -> NoReturn:
    logger.error("MongoDB %s failed (%s).", operation, type(error).__name__)
    raise OrderStoreError("Order storage is temporarily unavailable.") from None


def ping_mongodb() -> bool:
    """Verify MongoDB connectivity without changing stored data."""
    if db is None:
        raise OrderStoreError("MongoDB is not configured or unavailable.")
    response = db.command("ping")
    return response.get("ok") == 1


# ============================================================
# CREATE ORDER
# ============================================================

def create_order(order_items, total):
    order = {
        "items": order_items,
        "total": total,
        "status": "pending",
        "created_at": datetime.now(UTC),
    }

    try:
        result = _require_orders_collection().insert_one(order)
    except PyMongoError as error:
        _raise_storage_error("create", error)

    if result.inserted_id is None:
        logger.error("MongoDB create returned no order identifier.")
        raise OrderStoreError("Order storage did not confirm order creation.")

    return str(result.inserted_id)


# ============================================================
# GET ORDER
# ============================================================

def get_order(order_id: str):
    if not isinstance(order_id, str) or not ObjectId.is_valid(order_id):
        return None

    try:
        return _require_orders_collection().find_one(
            {"_id": ObjectId(order_id)}
        )
    except PyMongoError as error:
        _raise_storage_error("lookup", error)


# ============================================================
# CANCEL ORDER
# ============================================================

def cancel_order(order_id: str):
    if not isinstance(order_id, str) or not ObjectId.is_valid(order_id):
        return False

    try:
        result = _require_orders_collection().update_one(
            {
                "_id": ObjectId(order_id),
                "status": "pending"
            },
            {
                "$set": {
                    "status": "cancelled",
                    "updated_at": datetime.now(UTC),
                }
            }
        )

        return result.modified_count > 0

    except PyMongoError as error:
        _raise_storage_error("cancellation", error)
# ============================================================
# MODIFY ORDER
# ============================================================

def modify_order(
    order_id: str,
    order_items: list,
    total: float
):
    if not isinstance(order_id, str) or not ObjectId.is_valid(order_id):
        return False

    try:
        result = _require_orders_collection().update_one(
            {
                "_id": ObjectId(order_id),
                "status": "pending"
            },
            {
                "$set": {
                    "items": order_items,
                    "total": total,
                    "updated_at": datetime.now(UTC),
                }
            }
        )

        return result.modified_count > 0

    except PyMongoError as error:
        _raise_storage_error("modification", error)
# ============================================================
# TEST CONNECTION
# ============================================================

if __name__ == "__main__":

    try:
        if ping_mongodb():
            print("MongoDB ping successful.")
        else:
            print("MongoDB ping returned an unsuccessful response.")
    except OrderStoreError:
        print("MongoDB is not configured or unavailable.")
    except OperationFailure as error:
        if error.code == 18:
            print("MongoDB authentication failed.")
        else:
            print("MongoDB rejected the ping command.")
    except ServerSelectionTimeoutError:
        print("MongoDB server selection timed out.")
    except PyMongoError as error:
        print(f"MongoDB ping failed ({type(error).__name__}).")