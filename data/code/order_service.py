"""
Order Service - Handles e-commerce order processing.
Author: Alex Johnson
Team: Backend Team
Dependencies: MongoDB, RabbitMQ, User Service, Inventory Service
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

from .config import settings
from .messaging import publish_event
from .auth import get_current_user

app = FastAPI(title="Order Service", version="1.0.0")

# MongoDB client
mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
db = mongo_client.orders_db


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    """Schema for order item."""
    product_id: str
    product_name: str
    quantity: int
    unit_price: float


class ShippingAddress(BaseModel):
    """Schema for shipping address."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str


class OrderCreate(BaseModel):
    """Schema for creating an order."""
    items: List[OrderItem]
    shipping_address: ShippingAddress
    payment_method: str


class Order(BaseModel):
    """Schema for order response."""
    id: str
    user_id: int
    items: List[OrderItem]
    shipping_address: ShippingAddress
    status: OrderStatus
    total_amount: float
    created_at: datetime
    updated_at: datetime


async def validate_inventory(items: List[OrderItem]) -> bool:
    """
    Validate inventory availability via Inventory Service.

    Calls Inventory Service API to check stock levels.
    Returns True if all items are available.
    """
    async with httpx.AsyncClient() as client:
        for item in items:
            response = await client.get(
                f"{settings.INVENTORY_SERVICE_URL}/api/v1/inventory/{item.product_id}"
            )
            if response.status_code != 200:
                return False

            stock = response.json()
            if stock["available_quantity"] < item.quantity:
                return False

    return True


async def reserve_inventory(order_id: str, items: List[OrderItem]) -> bool:
    """
    Reserve inventory for an order.

    Calls Inventory Service to lock stock for this order.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.INVENTORY_SERVICE_URL}/api/v1/inventory/reserve",
            json={
                "order_id": order_id,
                "items": [
                    {"product_id": item.product_id, "quantity": item.quantity}
                    for item in items
                ]
            }
        )
        return response.status_code == 200


async def release_inventory(order_id: str) -> bool:
    """Release reserved inventory when order is cancelled."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.INVENTORY_SERVICE_URL}/api/v1/inventory/release",
            json={"order_id": order_id}
        )
        return response.status_code == 200


def calculate_total(items: List[OrderItem]) -> float:
    """Calculate order total amount."""
    return sum(item.quantity * item.unit_price for item in items)


@app.post("/api/v1/orders", response_model=Order)
async def create_order(
    order_data: OrderCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new order.

    1. Validates inventory availability
    2. Reserves stock via Inventory Service
    3. Creates order record in MongoDB
    4. Publishes OrderCreated event to RabbitMQ
    5. Triggers notification via Notification Service
    """
    # Validate inventory
    if not await validate_inventory(order_data.items):
        raise HTTPException(
            status_code=400,
            detail="One or more items are out of stock"
        )

    # Create order document
    order_doc = {
        "user_id": current_user["id"],
        "items": [item.dict() for item in order_data.items],
        "shipping_address": order_data.shipping_address.dict(),
        "payment_method": order_data.payment_method,
        "status": OrderStatus.PENDING,
        "total_amount": calculate_total(order_data.items),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    # Insert into MongoDB
    result = await db.orders.insert_one(order_doc)
    order_id = str(result.inserted_id)

    # Reserve inventory
    if not await reserve_inventory(order_id, order_data.items):
        await db.orders.delete_one({"_id": result.inserted_id})
        raise HTTPException(
            status_code=500,
            detail="Failed to reserve inventory"
        )

    # Update status to confirmed
    await db.orders.update_one(
        {"_id": result.inserted_id},
        {"$set": {"status": OrderStatus.CONFIRMED}}
    )

    # Publish event to RabbitMQ
    background_tasks.add_task(
        publish_event,
        "order.created",
        {"order_id": order_id, "user_id": current_user["id"]}
    )

    order_doc["id"] = order_id
    order_doc["status"] = OrderStatus.CONFIRMED
    return Order(**order_doc)


@app.get("/api/v1/orders/{order_id}", response_model=Order)
async def get_order(
    order_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get order by ID."""
    from bson import ObjectId

    order = await db.orders.find_one({"_id": ObjectId(order_id)})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    order["id"] = str(order["_id"])
    return Order(**order)


@app.delete("/api/v1/orders/{order_id}")
async def cancel_order(
    order_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Cancel an order.

    Only allowed if order status is PENDING or CONFIRMED.
    Releases reserved inventory and publishes cancellation event.
    """
    from bson import ObjectId

    order = await db.orders.find_one({"_id": ObjectId(order_id)})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    if order["status"] not in [OrderStatus.PENDING, OrderStatus.CONFIRMED]:
        raise HTTPException(
            status_code=400,
            detail="Order cannot be cancelled"
        )

    # Release inventory
    await release_inventory(order_id)

    # Update order status
    await db.orders.update_one(
        {"_id": ObjectId(order_id)},
        {
            "$set": {
                "status": OrderStatus.CANCELLED,
                "updated_at": datetime.utcnow()
            }
        }
    )

    # Publish cancellation event
    background_tasks.add_task(
        publish_event,
        "order.cancelled",
        {"order_id": order_id, "user_id": current_user["id"]}
    )

    return {"message": "Order cancelled successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "order-service"}
