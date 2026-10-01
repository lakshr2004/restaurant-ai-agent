from langgraph.graph import StateGraph, START, END

from app.graph.state import RestaurantState

from app.graph.nodes import (
    scope_guard,
    intent_router,
    menu_node,
    order_node,
    confirmation_node,
    order_status_node,
    order_cancel_node,
    order_modify_node,
    restaurant_info_node,
    out_of_scope_node,
)


# ============================================================
# ROUTE AFTER SCOPE
# ============================================================

def route_after_scope(state: RestaurantState):

    if state["scope"] == "RESTAURANT":
        return "restaurant"

    return "out_of_scope"


# ============================================================
# ROUTE AFTER INTENT
# ============================================================

def route_after_intent(state: RestaurantState):

    intent = state["intent"]

    if intent == "MENU_QUERY":
        return "menu"

    if intent == "ORDER_CREATE":
        return "order"

    if intent == "ORDER_STATUS":
        return "order_status"

    if intent == "ORDER_CANCEL":
        return "order_cancel"

    if intent == "ORDER_MODIFY":
        return "order_modify"

    if intent == "RESTAURANT_INFO":
        return "restaurant_info"

    return "out_of_scope"


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(RestaurantState)


# ============================================================
# NODES
# ============================================================

builder.add_node(
    "scope_guard",
    scope_guard
)

builder.add_node(
    "intent_router",
    intent_router
)

builder.add_node(
    "menu",
    menu_node
)

builder.add_node(
    "order",
    order_node
)

builder.add_node(
    "confirmation",
    confirmation_node
)

builder.add_node(
    "order_status",
    order_status_node
)

builder.add_node(
    "order_cancel",
    order_cancel_node
)

builder.add_node(
    "order_modify",
    order_modify_node
)

builder.add_node(
    "restaurant_info",
    restaurant_info_node
)

builder.add_node(
    "out_of_scope",
    out_of_scope_node
)


# ============================================================
# START
# ============================================================

builder.add_edge(
    START,
    "scope_guard"
)


# ============================================================
# SCOPE ROUTING
# ============================================================

builder.add_conditional_edges(
    "scope_guard",
    route_after_scope,
    {
        "restaurant": "intent_router",
        "out_of_scope": "out_of_scope",
    },
)


# ============================================================
# INTENT ROUTING
# ============================================================

builder.add_conditional_edges(
    "intent_router",
    route_after_intent,
    {
        "menu": "menu",

        "order": "order",

        "order_status": "order_status",

        "order_cancel": "order_cancel",

        "order_modify": "order_modify",

        "restaurant_info": "restaurant_info",

        "out_of_scope": "out_of_scope",
    },
)


# ============================================================
# ORDER CREATE FLOW
# ============================================================

builder.add_edge(
    "order",
    END
)


# ============================================================
# END NODES
# ============================================================

builder.add_edge(
    "menu",
    END
)

builder.add_edge(
    "confirmation",
    END
)

builder.add_edge(
    "order_status",
    END
)

builder.add_edge(
    "order_cancel",
    END
)

builder.add_edge(
    "order_modify",
    END
)

builder.add_edge(
    "restaurant_info",
    END
)

builder.add_edge(
    "out_of_scope",
    END
)


# ============================================================
# COMPILE
# ============================================================

graph = builder.compile()