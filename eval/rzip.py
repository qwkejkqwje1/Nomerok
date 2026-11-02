import io, urllib.request, zipfile
class RF(io.RawIOBase):
    def __init__(s, url):
        s.url = url; s.pos = 0
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60); s.size = int(r.headers['Content-Length'])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else s.pos + off if wh == 1 else s.size + off; return s.pos
    def readinto(s, b):
        n = len(b)
        if s.pos >= s.size or n == 0: return 0
        end = min(s.pos + n, s.size) - 1
        for a in range(5):
            try:
                d = urllib.request.urlopen(urllib.request.Request(s.url, headers={'Range': f'bytes={s.pos}-{end}'}), timeout=120).read(); break
            except Exception as ex:
                import time; time.sleep(3)
        b[:len(d)] = d; s.pos += len(d); return len(d)
def open_zip(url, buf=1 << 16):
    return zipfile.ZipFile(io.BufferedReader(RF(url), buffer_size=buf))
