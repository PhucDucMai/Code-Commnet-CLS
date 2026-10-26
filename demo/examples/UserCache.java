package com.example.cache;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Caches {@link User} objects in memory to avoid repeated database lookups.
 * Use {@link #get(String)} to look up a user by id, for example
 * {@code cache.get("u-42")}.
 * <p>
 * Entries are never evicted, because the number of users is small and
 * the cache is rebuilt on every deployment.
 *
 * @author Jane Doe
 * @see UserRepository
 * @deprecated use {@link RedisUserCache} instead, this class will be removed in 3.0.
 */
@Deprecated
public class UserCache {

    private final Map<String, User> users = new ConcurrentHashMap<>();

    /** Returns the cached user, or null if absent. */
    public User get(String id) {
        return users.get(id);
    }

    /**
     * Thread-safe holder for cache statistics.
     * Call {@link #reset()} before each benchmark run.
     */
    static final class Stats {
        long hits;
        long misses;

        void reset() {
            hits = 0;
            misses = 0;
        }
    }
}
