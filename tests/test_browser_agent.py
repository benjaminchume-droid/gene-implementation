from gene.browser.service import BrowserService


def test_browser_start_and_navigate():

    browser = BrowserService()

    try:
        result = browser.start(
            headless=True
        )

        assert result["success"] is True

        result = browser.navigate(
            "data:text/html,"
            "<html><head><title>Gene Test</title>"
            "</head><body>"
            "<h1>Browser Runtime Test</h1>"
            "<p>Gene browser is operational.</p>"
            "</body></html>"
        )

        assert result["success"] is True

        status = browser.status()

        assert status["running"] is True

    finally:
        browser.stop()
