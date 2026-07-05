"""Truth-check: does cognee 1.2.2 enforce per-user memory isolation?

The high-level remember/recall take no `user`; add/cognify/search do. This probe
flips ENABLE_BACKEND_ACCESS_CONTROL on, creates two users, ingests a fact as user
A, and checks that user B cannot retrieve it while user A can. The result decides
how Aeg wires real access control (Feature 4).

    ENABLE_BACKEND_ACCESS_CONTROL=true uv run python scripts/verify_access_control.py
"""

import asyncio
import os

# access control must be enabled BEFORE cognee is imported.
os.environ.setdefault("AEG_SCRATCH_DIR", "/tmp/aeg_ac_probe")
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "true"
os.environ.setdefault("REQUIRE_AUTHENTICATION", "false")

from aeg import cognee_client  # noqa: E402  (applies cognee env, then imports cognee)

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402
from cognee.modules.users.methods import create_user, get_user_by_email  # noqa: E402

FACT = "Project Atlas uses Postgres as its primary database."
QUERY = "What database does Project Atlas use? Answer in one word."


async def _get_or_create(email: str):
    try:
        u = await get_user_by_email(email)
        if u:
            return u
    except Exception:
        pass
    return await create_user(email=email, password=f"pw-{email}", is_verified=True)


async def main() -> None:
    print(f"access_control_env = {os.environ.get('ENABLE_BACKEND_ACCESS_CONTROL')}")
    await cognee_client._ensure_schema()  # create users/principals/… on a fresh pg
    a = await _get_or_create("owner-a@example.com")
    b = await _get_or_create("intruder-b@example.com")
    print(f"user A = {a.id}\nuser B = {b.id}")

    ds = "ac_probe_a"
    await cognee.add(FACT, dataset_name=ds, user=a)
    await cognee.cognify(datasets=[ds], user=a)
    print("ingested FACT as user A")

    async def _search(user, label):
        try:
            r = await cognee.search(query_text=QUERY, query_type=SearchType.GRAPH_COMPLETION,
                                    user=user, datasets=[ds])
            txt = " ".join(str(x) for x in (r or []))[:160]
            print(f"  {label}: {txt!r}  -> postgres_mentioned={'postgres' in txt.lower()}")
            return "postgres" in txt.lower()
        except Exception as e:
            print(f"  {label}: BLOCKED by error {type(e).__name__}: {str(e)[:120]}")
            return False

    print("search results:")
    a_sees = await _search(a, "A (owner, should SEE)")
    b_sees = await _search(b, "B (intruder, should NOT see)")

    print("\nVERDICT:")
    if a_sees and not b_sees:
        print("  ISOLATION ENFORCED — A sees its data, B does not. Access control works.")
    elif a_sees and b_sees:
        print("  NOT ISOLATED — B can read A's data (access control not enforced here).")
    else:
        print("  INCONCLUSIVE — A could not read its own data; check backend/setup.")


if __name__ == "__main__":
    asyncio.run(main())
