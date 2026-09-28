# AI Shopping Assistant

## The idea

You tell the assistant what you want:

> "Buy me running shoes under €120, delivered by Friday."

It knows your tastes, searches several stores, picks the best option, asks for your OK, and pays.

## How it's built

One **LangGraph agent**. The four chapters below are all parts of this agent.
The agent calls a few **external services** that sit outside it: the memory database, the stores, and the payment services.

```
User ──► Personal Memory ──► Product Search ──► Decision & Human Approval ──► Payment
              │                    │                       │                     │
         Memory DB           Store agents                 You             Payment services
```

## Chapter 1 — Personal Memory

The assistant remembers everything you bought and returned, and learns your preferences from it: sizes, colours, brands, price range, and what didn't work.

- Loads your preferences at the start of each request
- Updates them after each purchase or return
- Can explain why it thinks you like something

**Tech:** PostgreSQL, Qdrant, LangGraph memory

## Chapter 2 — Product Search

The assistant searches several stores at the same time and compares the results.

- Asks each store for matching products, prices, stock, and delivery dates
- Reads customer reviews to find each product's real pros and cons
- Checks each store's return and warranty rules
- Ranks the options and explains its choice

**Tech:** A2A, MCP, hybrid RAG (reviews and store policies)

## Chapter 3 — Decision & Human Approval

The assistant proposes a purchase and waits for your OK.

- Double-checks its recommendation before showing it
- Shows you the product, the store, the price, and why
- Does nothing until you approve, change, or cancel

**Tech:** LangGraph `interrupt()`

## Chapter 4 — Payment

The assistant pays, only within limits you set.

- You set a spending limit in advance
- It can never go over it
- It completes the checkout with the store

**Tech:** AP2 (spending limits), ACP (checkout), Stripe test mode

## External services

- **Memory database:** stores your purchase history and preferences
- **Test stores:** Shopify development stores and a few simulated stores we build ourselves
- **Payment services:** Stripe test mode, so no real money is involved

## Build order

1. **Basic version:** a very simple version of all 4 chapters, working end to end
2. **Product Search:** in depth
3. **Decision & Human Approval:** in depth
4. **Payment:** in depth

**Personal Memory** is built in parallel by the second person and plugged in when ready.

## Stack

Python, FastAPI, LangGraph, PostgreSQL, Qdrant, Next.js, Docker
