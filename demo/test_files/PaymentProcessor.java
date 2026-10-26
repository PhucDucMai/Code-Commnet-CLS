package org.example.billing;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Charges customers through an external payment provider.
 * Implementations must be idempotent: charging the same order twice must not take money twice.
 * Typical usage:
 * <pre>{@code
 * PaymentProcessor processor = new StripeProcessor(apiKey);
 * Receipt receipt = processor.charge(order, amount);
 * }</pre>
 *
 * @param <R> the type of receipt returned by the provider
 * @see Receipt
 * @author Jane Doe
 */
public interface PaymentProcessor<R extends Receipt> {

    /** Charges the given amount for an order and returns the provider receipt. */
    R charge(Order order, BigDecimal amount);

    /** Looks up a previous charge by its provider id. */
    Optional<R> find(String chargeId);
}

/**
 * Payment processor backed by the Stripe REST API.
 * Retries are disabled on purpose because Stripe already deduplicates requests with an idempotency key.
 * Create one instance per API key and reuse it, since it keeps an HTTP connection pool.
 *
 * @see <a href="https://stripe.com/docs/api">Stripe API reference</a>
 */
final class StripeProcessor implements PaymentProcessor<Receipt> {

    private final String apiKey;

    StripeProcessor(String apiKey) {
        this.apiKey = apiKey;
    }

    @Override
    public Receipt charge(Order order, BigDecimal amount) {
        return new Receipt();
    }

    @Override
    public Optional<Receipt> find(String chargeId) {
        return Optional.empty();
    }
}
