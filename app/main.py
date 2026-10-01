import logging
from typing import cast

from app.graph.graph import graph
from app.graph.nodes import (
    confirmation_node,
    cancellation_confirmation_node,
    LLMServiceError,
)
from app.graph.state import RestaurantState
from app.tools.menu import MenuDataError
from app.tools.orders import OrderStoreError


logger = logging.getLogger(__name__)

EMPTY_INPUT_RESPONSE = "Please enter a message so I can help."
LLM_FAILURE_RESPONSE = (
    "Sorry, I'm having trouble processing your request right now. "
    "Please try again in a moment."
)
ORDER_FAILURE_RESPONSE = (
    "Sorry, I'm having trouble accessing the order system right now. "
    "Please try again in a moment."
)
MENU_FAILURE_RESPONSE = (
    "Sorry, our menu is temporarily unavailable. Please try again in a moment."
)
GENERAL_FAILURE_RESPONSE = (
    "Sorry, something went wrong while handling your request. "
    "Please try again in a moment."
)

def run_agent(
    user_message: str,
    state: RestaurantState,
) -> RestaurantState:
    state = {
        **state,
        "user_message": user_message,
        "scope": "",
        "intent": "",
        "menu_query": "",
        "response": "",
    }

    if not isinstance(user_message, str) or not user_message.strip():
        return {
            **state,
            "response": EMPTY_INPUT_RESPONSE,
        }

    try:
        # Pending order confirmation
        if state.get("pending_order_confirmation", False):
            result = confirmation_node(state)

        # Pending cancellation confirmation
        elif state.get("pending_cancellation_confirmation", False):
            result = cancellation_confirmation_node(state)

        # Normal graph flow
        else:
            result = cast(RestaurantState, graph.invoke(state))

        return result

    except LLMServiceError:
        return {
            **state,
            "response": LLM_FAILURE_RESPONSE,
        }
    except OrderStoreError as error:
        logger.error("Order request failed (%s).", type(error).__name__)
        return {
            **state,
            "response": ORDER_FAILURE_RESPONSE,
        }
    except MenuDataError as error:
        logger.error("Menu request failed (%s).", type(error).__name__)
        return {
            **state,
            "response": MENU_FAILURE_RESPONSE,
        }
    except Exception as error:
        logger.error("Agent request failed (%s).", type(error).__name__)
        return {
            **state,
            "response": GENERAL_FAILURE_RESPONSE,
        }


if __name__ == "__main__":

    state: RestaurantState = {
        "user_message": "",
        "response": "",
        "scope": "",
        "intent": "",
        "menu_query": "",
        "current_order": None,
        "order_id": "",
        "pending_order_confirmation": False,
        "pending_cancellation_confirmation": False,
    }

    while True:

        try:
            user_message = input("Customer: ")
        except EOFError:
            print("Agent: Goodbye! 👋")
            break

        if user_message.lower() in {
            "exit",
            "quit",
        }:
            print("Agent: Goodbye! 👋")
            break

        result = run_agent(
            user_message,
            state
        )

        # Save updated state
        state = result

        print(
            "Agent:",
            result["response"]
        )

        print()