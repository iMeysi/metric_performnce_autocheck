"""OS-specific primitives; failures must not silently disable locking/accounting."""
import os
import sys


def acquire_lock(path):
    handle = open(str(path), 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle  # Closing the descriptor releases the OS lock.
    except BaseException:
        handle.close()
        raise


def peak_rss_bytes():
    if os.name == 'nt':
        import psutil
        return int(psutil.Process().memory_info().peak_wset)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)
