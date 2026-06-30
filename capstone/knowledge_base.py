"""
The store's official policies. In a real company this would be thousands of
documents pulled from a help center / CMS -- which is exactly why you need a
scalable CLOUD vector database (Pinecone) instead of a local list.

Each policy has an id (so the agent can CITE which policy it used) and text.
"""

STORE_POLICIES = [
    {"id": "shipping-standard",
     "text": "Standard shipping is free on orders over $50 and takes 3 to 5 business days within the continental US."},
    {"id": "shipping-express",
     "text": "Express shipping costs $15 and delivers within 1 to 2 business days. Order before 2 PM for same-day dispatch."},
    {"id": "shipping-international",
     "text": "International shipping is available to over 60 countries. Delivery takes 7 to 14 business days and duties are paid by the customer."},
    {"id": "returns-window",
     "text": "Items can be returned within 30 days of delivery for a full refund, as long as they are unused and in original packaging."},
    {"id": "returns-electronics",
     "text": "Opened electronics can be returned within 14 days but are subject to a 15 percent restocking fee unless the item is defective."},
    {"id": "refund-timing",
     "text": "Refunds are processed within 5 to 7 business days after we receive the returned item, back to the original payment method."},
    {"id": "warranty",
     "text": "All electronics come with a 1 year manufacturer warranty covering defects. Accidental damage is not covered."},
    {"id": "warranty-extended",
     "text": "An optional 3 year extended warranty can be purchased at checkout for 10 percent of the product price."},
    {"id": "payment-methods",
     "text": "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay. We do not accept cash on delivery."},
    {"id": "order-cancel",
     "text": "Orders can be cancelled free of charge within 1 hour of placement. After that, the order may have already shipped."},
    {"id": "order-tracking",
     "text": "Once an order ships, a tracking number is emailed to you and can also be found under My Orders in your account."},
    {"id": "price-match",
     "text": "We price match identical in-stock items from major authorized retailers within 7 days of your purchase."},
    {"id": "damaged-item",
     "text": "If an item arrives damaged, report it within 48 hours with photos and we will send a free replacement immediately."},
    {"id": "account-help",
     "text": "You can reset your password from the login page. For account lockouts, contact support and verify your identity."},
]
