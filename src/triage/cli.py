"""Account provisioning: health organizations create accounts, nobody signs up alone.

Usage:
  uv run python -m triage.cli create-user --email ana@example.com --name "Ana" --role promoter
The password is asked interactively, or read from the TRIAGE_PASSWORD variable.
"""

import argparse
import asyncio
import getpass
import os
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from triage.config import get_settings
from triage.db.models import User, UserRole
from triage.db.session import create_engine
from triage.security import hash_password


async def create_user(email: str, name: str, role: UserRole, password: str) -> None:
    engine = create_engine(get_settings().database_url)
    try:
        async with async_sessionmaker(engine)() as session:
            email = email.strip().lower()
            if await session.scalar(select(User).where(User.email == email)):
                print(f"User {email} already exists", file=sys.stderr)
                raise SystemExit(1)
            session.add(
                User(
                    email=email,
                    full_name=name.strip(),
                    role=role,
                    password_hash=await hash_password(password),
                )
            )
            await session.commit()
            print(f"Created {role.value} {email}")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="triage.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user", help="Create a promoter or coordinator account")
    create.add_argument("--email", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--role", required=True, choices=[role.value for role in UserRole])
    args = parser.parse_args()

    password = os.environ.get("TRIAGE_PASSWORD") or getpass.getpass("Password: ")
    if len(password) < 10:
        parser.error("The password must have at least 10 characters")
    asyncio.run(create_user(args.email, args.name, UserRole(args.role), password))


if __name__ == "__main__":
    main()
