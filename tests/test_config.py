from triage.config import normalize_database_url


def test_neon_urls_are_adapted_to_asyncpg() -> None:
    url = "postgresql://u:p@ep-x.us-east-1.aws.neon.tech/db?sslmode=require&channel_binding=require"

    assert (
        normalize_database_url(url)
        == "postgresql+asyncpg://u:p@ep-x.us-east-1.aws.neon.tech/db?ssl=require"
    )
