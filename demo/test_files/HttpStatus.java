package org.example.http;

/**
 * Standard HTTP status codes used by the web server.
 * Use {@link #fromCode(int)} to turn a numeric code into a constant.
 *
 * @see <a href="https://www.rfc-editor.org/rfc/rfc9110">RFC 9110</a>
 */
@SuppressWarnings({"unused", "java:S115"})
public enum HttpStatus {
    OK(200), NOT_FOUND(404), INTERNAL_ERROR(500);

    private final int code;

    HttpStatus(int code) {
        this.code = code;
    }

    /** Returns the constant for a numeric code, or {@code INTERNAL_ERROR} if it is unknown. */
    public static HttpStatus fromCode(int code) {
        for (HttpStatus s : values()) {
            if (s.code == code) {
                return s;
            }
        }
        return INTERNAL_ERROR;
    }

    /**
     * Groups status codes into families such as 2xx success and 4xx client errors.
     * The family is derived from the first digit, so non-standard codes still get a family.
     */
    public static final class Family {
        private Family() {
        }
    }
}

/**
 * Immutable range of status codes, from {@code from} to {@code to} inclusive.
 * Ranges are compared by their lower bound.
 */
record StatusRange(int from, int to) {
}
