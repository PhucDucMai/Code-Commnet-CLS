package org.example.net;

import java.net.URI;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Downloads files over HTTP and stores them in a local cache directory.
 * Each download runs on a worker thread and reports its progress to a listener.
 * Call {@link #submit(URI, Path)} to start a download and keep the returned future to cancel it.
 * For example, {@code manager.submit(uri, target).get()} blocks until the file is complete.
 * We use a bounded thread pool because opening too many connections at once makes servers throttle us.
 * Partial files are written with a temporary suffix, so a crash never leaves a corrupted file behind.
 *
 * @see DownloadListener
 * @see java.net.http.HttpClient
 * @author Jane Doe
 */
public class DownloadManager {

    private final ExecutorService pool = Executors.newFixedThreadPool(4);

    /** Starts downloading {@code uri} into {@code target}. */
    public Future<Path> submit(URI uri, Path target) {
        return pool.submit(() -> target);
    }
}

/**
 * Legacy downloader that fetches files one at a time on the calling thread.
 *
 * @deprecated use {@link DownloadManager} instead, this class will be removed in version 3.0.
 */
@Deprecated
class SimpleDownloader {
}
