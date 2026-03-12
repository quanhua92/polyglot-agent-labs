use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub content: String,
    pub category: String,
}

pub fn load_documents() -> Vec<Document> {
    vec![
        Document {
            id: "password-reset".to_string(),
            title: "How to Reset Your Password".to_string(),
            category: "account".to_string(),
            content: "To reset your password:
1. Go to the login page and click 'Forgot Password'
2. Enter your email address
3. Check your email for the reset link (valid for 24 hours)
4. Create a new password (min 8 characters, must include uppercase, lowercase, and number)
5. Log in with your new password

If you don't receive the email within 5 minutes:
- Check your spam/junk folder
- Verify you're using the correct email address
- Contact support at help@example.com".to_string(),
        },
        Document {
            id: "account-creation".to_string(),
            title: "Creating a New Account".to_string(),
            category: "account".to_string(),
            content: "To create a new account:
1. Click 'Sign Up' on the homepage
2. Enter your email address and create a password
3. Verify your email by clicking the link sent to you
4. Complete your profile information (name, phone, address)
5. Agree to the Terms of Service

Benefits of having an account:
- Faster checkout with saved addresses
- Order tracking and history
- Exclusive member discounts
- Wishlist functionality".to_string(),
        },
        Document {
            id: "order-tracking".to_string(),
            title: "Track Your Order Status".to_string(),
            category: "shipping".to_string(),
            content: "Track your order in several ways:

1. Email Confirmation:
   - Find your tracking number in the shipping confirmation email
   - Click the tracking link to see real-time updates

2. Account Dashboard:
   - Log into your account
   - Go to 'Order History'
   - Click 'Track Order' next to your order

3. Order Status:
   - Processing: Order is being prepared
   - Shipped: Package is in transit
   - Delivered: Package has been delivered
   - Delayed: Shipping delay occurred

For issues, contact shipping@example.com".to_string(),
        },
        Document {
            id: "return-policy".to_string(),
            title: "Return Policy and Process".to_string(),
            category: "returns".to_string(),
            content: "Our 30-day return policy:

Eligible Items:
- Unused items in original packaging
- Items returned within 30 days of delivery
- Non-final sale items

Return Process:
1. Log into your account > Order History
2. Select the item(s) to return
3. Print the prepaid return label
4. Pack items securely in original packaging
5. Drop off at any authorized location

Refund Timeline:
- Processing: 3-5 business days after receipt
- Credit to original payment method: 5-7 business days

Non-returnable:
- Personalized/custom items
- Perishable goods
- Gift cards
- Final sale items".to_string(),
        },
        Document {
            id: "payment-methods".to_string(),
            title: "Accepted Payment Methods".to_string(),
            category: "billing".to_string(),
            content: "We accept the following payment methods:

Credit/Debit Cards:
- Visa, MasterCard, American Express, Discover
- Cards must be enabled for online purchases

Digital Wallets:
- PayPal
- Apple Pay
- Google Pay
- Samsung Pay

Security:
- All transactions are encrypted with SSL
- We never store your full card number
- 3D Secure authentication for enhanced security

Payment Issues:
- Ensure sufficient funds
- Verify billing address matches card
- Try alternative payment method
- Contact your bank if payment is declined

For billing questions: billing@example.com".to_string(),
        },
        Document {
            id: "shipping-options".to_string(),
            title: "Shipping Options and Delivery".to_string(),
            category: "shipping".to_string(),
            content: "Shipping options and pricing:

Standard Shipping (5-7 business days): $4.99
Express Shipping (2-3 business days): $9.99
Overnight Shipping (1 business day): $19.99

FREE standard shipping on orders over $50!

Shipping Guidelines:
- Orders before 2 PM EST ship same day
- Business days only (Monday-Friday)
- No weekend or holiday delivery

Shipping Destinations:
- All 50 US states
- PO Boxes and APO/FPO addresses
- International shipping not available

Delivery Issues:
- Delayed due to weather or holidays
- You'll receive email notifications
- Contact shipping@example.com for urgent issues".to_string(),
        },
        Document {
            id: "profile-update".to_string(),
            title: "Update Account Profile".to_string(),
            category: "account".to_string(),
            content: "How to update your profile:

1. Log into your account
2. Click 'Account Settings'
3. Update any of the following:
   - Email address (requires verification)
   - Password
   - Phone number
   - Shipping addresses
   - Communication preferences

Changes save automatically.

Email Change:
- New email requires verification
- Old email will receive notification
- Update all saved logins

Password Update:
- Current password required
- New password must meet security requirements
- Confirmation required

Address Book:
- Save multiple addresses
- Set default shipping address
- Add billing addresses".to_string(),
        },
        Document {
            id: "cancel-order".to_string(),
            title: "Cancel or Modify an Order".to_string(),
            category: "orders".to_string(),
            content: "Order Cancellation:

Before Shipping:
- Log into Account > Order History
- Click 'Cancel Order' if available
- Immediate cancellation is usually possible
- Refund to original payment method

After Shipping:
- Cannot cancel once shipped
- Follow return process upon delivery
- Return shipping is free for most items

Order Modification:
- Address changes: Contact us immediately
- Item changes: Cancel and reorder
- Quantity changes: Cancel and reorder

Contact Options:
- Live chat: Available 24/7
- Email: orders@example.com
- Phone: 1-800-ORDERS

Act quickly—orders process within 1-2 hours!".to_string(),
        },
        Document {
            id: "technical-support".to_string(),
            title: "Technical Support and Troubleshooting".to_string(),
            category: "support".to_string(),
            content: "Common technical issues:

Website Problems:
- Clear browser cache and cookies
- Try a different browser (Chrome, Firefox, Safari)
- Disable browser extensions temporarily
- Ensure JavaScript is enabled
- Check internet connection

Mobile App:
- Update to latest version
- Force close and reopen the app
- Clear app cache (Settings > App Info)
- Reinstall if problems persist

Checkout Issues:
- Ensure all fields are completed
- Try different payment method
- Check card security code is correct
- Verify billing address matches card

Account Access:
- Reset password if needed
- Check caps lock on email/password
- Clear browser cookies
- Try incognito/private mode

Still having issues? Contact tech@example.com".to_string(),
        },
        Document {
            id: "loyalty-program".to_string(),
            title: "Loyalty Rewards Program".to_string(),
            category: "general".to_string(),
            content: "Our Loyalty Rewards Program:

Earning Points:
- 1 point per $1 spent
- 100 points = $5 reward
- Bonus points on special promotions
- Double points on your birthday

Member Tiers:
- Bronze: 0-999 points (5% cashback)
- Silver: 1000-4999 points (10% cashback)
- Gold: 5000+ points (15% cashback)

Redeeming Points:
- Apply at checkout
- No minimum redemption
- Points valid for 1 year
- Can combine with other offers

Exclusive Benefits:
- Early access to sales
- Free shipping on all orders
- Member-only discounts
- Birthday gift

Join free at checkout or in your account settings!".to_string(),
        },
        Document {
            id: "gift-cards".to_string(),
            title: "Gift Cards and Promotional Codes".to_string(),
            category: "billing".to_string(),
            content: "Gift Cards:

Purchasing:
- Available in $25, $50, $100, $200 denominations
- Digital delivery via email
- Physical cards available by mail
- No expiration date
- No fees or deductions

Redeeming:
- Enter code at checkout
- Check balance online or in-store
- Can be combined with other payment methods
- Partial redemption allowed (keep balance)

Promotional Codes:
- Enter in promo code field at checkout
- One code per order
- Cannot combine codes
- Check expiration dates
- Some exclusions may apply

Lost/Stolen Cards:
- Contact support with receipt
- Physical cards can be replaced
- Digital cards resend to email

Issue? Contact: billing@example.com".to_string(),
        },
    ]
}
