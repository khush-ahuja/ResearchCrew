import httpx

from researchcrew.tools import HttpFetch

HTML = """<html><head><style>body{}</style><script>track()</script></head><body>
<nav>Home | About</nav><h1>Vector databases</h1>
<p>Qdrant   stores   vectors.</p>


<p>HNSW is an index.</p><footer>(c) 2026</footer></body></html>"""


def test_http_fetch_keeps_readable_text_only(monkeypatch):
    def fake_get(url, **kw):
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    text = HttpFetch()("https://x.example")
    assert "Qdrant stores vectors." in text and "HNSW is an index." in text
    assert not any(junk in text for junk in ("track()", "Home | About", "(c) 2026", "body{}"))
    assert "\n\n\n" not in text


def test_http_fetch_truncates(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        200, text="a" * 100, headers={"content-type": "text/plain"}, request=httpx.Request("GET", url)))
    assert HttpFetch(max_chars=10)("https://x.example") == "a" * 10
