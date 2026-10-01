# Restaurant AI Agent

A backend-focused conversational restaurant agent built with Python, LangGraph, Groq, and MongoDB. The project is designed around a simple but important principle: the LLM handles natural-language understanding, while deterministic Python logic remains responsible for validation, pricing, availability, confirmation gates, and order persistence.

## Highlights

- Stateful conversational ordering
- LangGraph workflow orchestration
- Groq-powered language understanding
- Multi-item order extraction and validation
- Canonical menu-based pricing and availability checks
- Confirmation-before-persistence flow
- MongoDB-backed order lifecycle
- Status and cancellation workflows
- Defensively handled malformed LLM output
- 46 deterministic regression tests
- GitHub Actions CI

## Problem

Restaurant ordering conversations are easy for an LLM to misunderstand if the system blindly trusts extracted item names, quantities, or prices. This project addresses that by separating language understanding from business logic: the LLM can interpret intent, but Python validates the requested items against canonical menu data before anything is staged or persisted.

## Demo

### 1. Natural-language multi-item ordering

![Natural-language multi-item ordering](assets/Screenshot%202026-10-01%20142855.png)

The agent converts a natural-language request into a structured multi-item order, validates the request against canonical menu data, calculates the total, and asks for confirmation before persistence.

### 2. Confirmation before persistence

![Order confirmation before persistence](assets/Screenshot%202026-10-01%20142954.png)

The staged order is only written to MongoDB after the user explicitly confirms it. A `no` response discards the staged order instead of mutating the database.

### 3. Status and cancellation confirmation

![Order status and cancellation prompt](assets/Screenshot%202026-10-01%20143057.png)

The agent resolves the active order, displays the current status and items, and then asks for explicit confirmation before cancelling it.

### 4. Final cancelled state

![Cancelled order state](assets/Screenshot%202026-10-01%20143154.png)

After confirmation, the order is updated to a cancelled state and the final status is retrieved and displayed again for verification.

### 5. Automated validation

![Automated validation results](assets/Screenshot%202026-10-01%20143303.png)

The project includes a deterministic regression suite covering order creation, cancellation, validation failures, MongoDB safety, and state handling. The automated checks also validate syntax and dependency health.

## Architecture

```mermaid
flowchart TD
    A[Customer] --> B[CLI]
    B --> C[run_agent]
    C --> D{Confirmation pending?}
    D -- Yes --> E[Order or cancellation confirmation handler]
    D -- No --> F[LangGraph]

    F --> G[Scope Guard]
    G --> H{In restaurant scope?}
    H -- No --> I[Out-of-Scope Node]
    H -- Yes --> J[Intent Router]

    J --> K[Menu Node]
    J --> L[Order Node]
    J --> M[Order Status Node]
    J --> N[Order Modify Node]
    J --> O[Order Cancel Node]
    J --> P[Restaurant Info Node]

    K --> Q[Menu Tool]
    L --> Q
    N --> Q
    Q --> R[Canonical menu.json]

    L --> S[Order Tools]
    M --> S
    N --> S
    O --> S
    S --> T[(MongoDB)]

    G -. classification .-> U[Groq via LangChain Groq]
    J -. classification .-> U
    K -. extraction and response .-> U
    L -. item and quantity extraction .-> U
    N -. item and quantity extraction .-> U
    P -. response generation .-> U

    I --> V[Response]
    E --> V
    K --> V
    L --> V
    M --> V
    N --> V
    O --> V
    P --> V
    V --> B
```

### Major Layers

- **CLI and `run_agent()`:** accepts customer messages, carries the conversation state across turns, dispatches pending confirmations, and converts service failures into safe responses.
- **LangGraph:** routes requests through the scope guard and intent router before dispatching to the right node.
- **Graph nodes:** use the LLM for intent classification and response generation, while Python remains responsible for validation, totals, and order-ID handling.
- **Tools and data:** `menu.py` reads and validates `menu.json`; `orders.py` owns MongoDB order operations. Restaurant facts come from `restaurant_info.py`.
- **State:** tracks the current request, scope, intent, staged order, active order ID, confirmation flags, and final response.

This separation is a core design strength: the LLM interprets language, while Python enforces business rules.

## Key Features

- Restaurant scope guard and intent classification
- Menu search over canonical JSON menu data
- Single-item and multi-item order creation
- Order summary and confirmation before persistence
- Order status lookup using the current conversation's order ID
- Single-item order modification
- Order cancellation with explicit confirmation
- Restaurant information queries
- Out-of-scope request handling
- Typed, state-driven conversation flow
- Menu, quantity, availability, and price validation
- Safe error responses for LLM, menu, and order-store failures
- MongoDB persistence through PyMongo
- Groq LLM integration through LangChain Groq
- 46 deterministic `unittest` regression tests
- GitHub Actions CI on pushes and pull requests

## Architecture

```mermaid
flowchart TD
	A[Customer] --> B[CLI]
	B --> C[run_agent]
	C --> D{Confirmation pending?}
	D -- Yes --> E[Order or cancellation confirmation handler]
	D -- No --> F[LangGraph]

	F --> G[Scope Guard]
	G --> H{In restaurant scope?}
	H -- No --> I[Out-of-Scope Node]
	H -- Yes --> J[Intent Router]

	J --> K[Menu Node]
	J --> L[Order Node]
	J --> M[Order Status Node]
	J --> N[Order Modify Node]
	J --> O[Order Cancel Node]
	J --> P[Restaurant Info Node]

	K --> Q[Menu Tool]
	L --> Q
	N --> Q
	Q --> R[Canonical menu.json]

	L --> S[Order Tools]
	M --> S
	N --> S
	O --> S
	S --> T[(MongoDB)]

	G -. classification .-> U[Groq via LangChain Groq]
	J -. classification .-> U
	K -. extraction and response .-> U
	L -. item and quantity extraction .-> U
	N -. item and quantity extraction .-> U
	P -. response generation .-> U

	I --> V[Response]
	E --> V
	K --> V
	L --> V
	M --> V
	N --> V
	O --> V
	P --> V
	V --> B
```

### Major Layers

- **CLI and `run_agent()`:** accepts customer messages, carries `RestaurantState` between turns, dispatches pending yes/no confirmations, and converts service failures into safe responses.
- **LangGraph:** runs the scope guard and intent router, then routes each new request to its relevant node.
- **Graph nodes:** use the LLM for language understanding and response generation, while keeping validation, totals, and order-ID resolution in Python.
- **Tools and data:** `menu.py` reads and validates `menu.json`; `orders.py` owns MongoDB order operations. Restaurant facts come from `restaurant_info.py`.
- **State:** retains the current message, scope, intent, staged order, most recent order ID, confirmation flags, and response.

Pending order and cancellation confirmations are handled by `run_agent()` on the following turn. Order creation is not sent to MongoDB while it is only staged.

## Order Lifecycle

1. The order node asks the LLM to extract requested item names and quantities.
2. Python matches every item to the canonical menu and checks availability and positive quantity.
3. Python takes IDs and prices from menu data and calculates each line amount and the total.
4. The agent presents a complete summary and waits for confirmation.
5. Only a `yes` response calls the MongoDB create operation. A `no` discards the staged order.

If extraction or validation fails for any item, the entire order is rejected; no partial order is staged or persisted.

### Multi-Item Example

Customer:

> I want 2 Cheese Veg Burgers and 1 Classic French Fries

The agent resolves the names, item IDs, availability, and prices from `menu.json`. With the current menu, the Python-calculated total is `2 × ₹179 + 1 × ₹99 = ₹457`. The complete summary is shown before any database write.

## Modification and Cancellation

- **Modification:** Single-item modification is currently supported; multi-item modification is not implemented. The current flow extracts one replacement item and quantity, validates it against the canonical menu, recalculates the total in Python, and updates a pending order.
- **Cancellation:** the agent first asks for confirmation. `no` leaves the order unchanged; `yes` changes the pending order's status to `cancelled`.
- **Status:** the agent resolves an explicit MongoDB ObjectId from the message or falls back to the latest order ID in conversation state. Stored order items and total are displayed.

## Error Handling

- LLM invocation failures or missing response text produce a retry message; diagnostic logs include only the operation and exception type.
- Invalid JSON or incomplete order extraction fails closed without staging or persisting an order.
- Unknown items, unavailable items, and non-positive quantities are rejected before persistence.
- Menu loading validates file access, JSON structure, required item fields, price/availability types, and unique IDs.
- Invalid ObjectIds are rejected without querying or mutating an order. Missing orders are reported as not found; MongoDB failures return an order-system retry response.
- Blank CLI input asks the user to enter a message. `exit`, `quit`, EOF, and `KeyboardInterrupt` retain normal exit behavior.

## Tech Stack

| Technology | Use |
| --- | --- |
| Python 3.14 | Application and regression tests |
| LangGraph | Request routing and graph execution |
| Groq / LangChain Groq | LLM access for classification, extraction, and responses |
| MongoDB | Order persistence |
| PyMongo | MongoDB client and BSON/ObjectId support |
| python-dotenv | Local environment configuration |
| `unittest` | Deterministic regression suite |
| GitHub Actions | Push and pull-request CI |

## Environment Variables

Set these variable names in a local `.env` file:

```text
GROQ_API_KEY
MONGODB_URI
MONGODB_DATABASE
```

Do not commit `.env`. It is ignored by Git. The application loads these values locally through `python-dotenv`.

## Security Considerations

This project intentionally avoids storing credentials in the repository. The local `.env` file is never committed, and the Git ignore configuration prevents secrets from being tracked. The README and demo screenshots should avoid exposing any connection strings, tokens, API keys, or personal data. Only non-sensitive, de-identified order IDs should be shown when illustrating the MongoDB-backed lifecycle.

## Installation

```powershell
git clone <REPOSITORY_URL>
cd restaurant-ai-agent
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
notepad .env
```

Add the required variable names and your local values to `.env`, then save and close the editor. Keep the file local.

## Run

```powershell
python -m app.main
```

Enter a restaurant request at the `Customer:` prompt. Enter `exit` or `quit` to leave.

## Example Conversation

```text
Customer: I want 2 Cheese Veg Burgers and 1 Classic French Fries
Agent: Here's your order summary ... Total: ₹457. Would you like me to place this order?

Customer: yes
Agent: Your order has been placed successfully! ...

Customer: what is the status of my order
Agent: Status: pending ...

Customer: modify my order to 2 Paneer Burgers
Agent: Your order has been modified successfully! ...

Customer: cancel my order
Agent: Are you sure you want to cancel order <order_id>? Reply 'yes' ...

Customer: yes
Agent: Your order has been cancelled successfully!

Customer: what are your timings?
Agent: [Restaurant hours from configured restaurant information]

Customer: explain binary search in Java
Agent: Sorry, I can only help with restaurant-related queries and orders.
```

The modification example replaces the pending order contents with one item; it does not demonstrate multi-item modification.

## Testing

Run the deterministic regression suite and syntax compilation:

```powershell
python -m unittest discover -s tests -v
python -m compileall app tests
```

The suite contains 46 tests. It exercises the real graph and application logic while stubbing Groq and MongoDB operations; it does not require service credentials or network access.

Real Groq and MongoDB connectivity and a controlled end-to-end order lifecycle have also been manually verified separately. Those checks are not part of the automated suite and require valid local environment variables. The test order was cancelled after verification; the application has no delete helper.

## CI

`.github/workflows/ci.yml` runs on `push` and `pull_request` using Ubuntu and Python 3.14. It upgrades pip, installs `requirements.txt`, runs `pip check`, compiles `app` and `tests`, and runs the complete unittest suite. The automated workflow does not need a real `.env`, Groq API key, or MongoDB server.

## Project Structure

```text
restaurant-ai-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── agents/
│   │   └── restaurant_agent.py
│   ├── data/
│   │   ├── menu.json
│   │   └── restaurant_info.py
│   ├── graph/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── services/
│   │   └── validation.py
│   ├── tools/
│   │   ├── billing.py
│   │   ├── inventory.py
│   │   ├── menu.py
│   │   └── orders.py
│   └── main.py
├── tests/
│   └── test_regression.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Design Decisions

- The LLM interprets natural language; Python owns canonical matching, validation, ObjectId handling, and price arithmetic.
- `menu.json` is the source of truth for item IDs, names, prices, and availability.
- An order is staged in conversation state and persisted only after explicit confirmation.
- LangGraph routes new messages; `run_agent()` handles pending confirmation replies using state.
- Invalid extraction and invalid menu data fail closed instead of creating incomplete or guessed orders.
- MongoDB stores orders; restaurant information remains local configuration data.

## Future Improvements

- Multi-item order modification
- REST API and web interface
- Authentication and authorization
- Payment integration and delivery tracking
- Structured observability and retry/backoff policies
- Richer conversational memory across sessions
