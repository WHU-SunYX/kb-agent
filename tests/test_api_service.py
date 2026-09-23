from kb_agent.api.service import create_app


def test_health():
    app = create_app()
    routes = [r.path for r in app.routes]
    assert "/health" in routes
    assert "/api/v1/search" in routes
