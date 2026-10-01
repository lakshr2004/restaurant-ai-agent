from typing import TypedDict


class OrderItem(TypedDict):
    item_id: str
    name: str
    quantity: int
    price: int | float


class CurrentOrder(TypedDict):
    items: list[OrderItem]
    total: int | float


class RestaurantState(TypedDict):
    user_message: str
    scope: str
    intent: str
    menu_query: str
    current_order: CurrentOrder | None
    order_id: str
    pending_order_confirmation: bool
    pending_cancellation_confirmation: bool
    response: str